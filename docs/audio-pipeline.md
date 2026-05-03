# Audio / VAD / Smart Turn 仕様メモ

`docs/app.js` の現在の実装をもとに、このデモで音声がどのように取り込まれ、VAD と Smart Turn に渡され、各種バッファがどう扱われるかを整理したメモです。

## 全体像

1. ブラウザのマイク入力を 1ch で取得する。
2. `AudioContext` の入力サンプルレートから `16 kHz` に逐次リサンプルする。
3. リサンプル済みサンプルを `vadQueue` に積み、`512 samples` ごとに VAD を実行する。
4. 各 VAD チャンクは、判定より先に Smart Turn 用バッファ `recentAudioChunks` に追加される。
5. `vadProb >= vadThreshold` になったら `speechActive = true` として発話中に入る。
6. 発話中に `vadProb < vadThreshold` へ落ちたら、そこで即推論せず `Smart Turn delay` だけ待つ。
7. 待機中に発話が再開しなければ、その時点まで `recentAudioChunks` にたまっている直近音声を `8 秒固定長` に整形して Smart Turn に渡す。
8. Smart Turn の確率値をしきい値で 2 値化し、`Complete` / `Incomplete` を決める。
9. `Complete` の場合だけ、推論完了時に `recentAudioChunks` を空にする。`Incomplete` の場合は保持して次回に引き継ぐ。

この実装では Smart Turn はストリーミング推論ではなく、`VAD OFF` 後に一定時間の静音が続いたときに 1 回だけ実行されます。

## 主要パラメータ

| 項目 | 値 | 意味 |
| --- | --- | --- |
| VAD / Smart Turn 用レート | `16000 Hz` | すべてのモデル入力は 16 kHz 前提 |
| VAD チャンク | `512 samples` | 約 `32 ms` ごとに VAD を回す |
| Smart Turn delay | 初期値 `200 ms` | `VAD OFF` 後、Smart Turn 実行前に待つ時間 |
| Smart Turn delay の範囲 | `0 ms` から `300 ms` | UI スライダーで `10 ms` 刻み調整 |
| Smart Turn 最大長 | `8 s` | `128000 samples` |
| Smart Turn フレーム数 | `800 frames` | `hop_length = 160` に対応 |
| VAD しきい値 | 初期値 `0.2` | `vadProb >= threshold` で発話扱い |
| turn しきい値 | 初期値 `0.8` | `probability >= threshold` で `Complete` |
| VAD 状態リセット | `5000 ms` | 長時間連続時の内部状態を定期リセット |

## 音声入力の仕様

### マイク取得

- `getUserMedia()` は `audio.channelCount = 1` で取得します。
- `echoCancellation`、`noiseSuppression`、`autoGainControl` はすべて `false` です。
- `video` は使いません。

つまり入力は「モノラル、生のマイク波形に近いもの」を前提にしています。

### AudioContext と ScriptProcessor

- `AudioContext` の実サンプルレートはブラウザ依存です。多くの環境では `48 kHz` ですが、固定ではありません。
- `createScriptProcessor(4096, 1, 1)` で `onaudioprocess` を受けます。
- `outputBuffer` は毎回 `0` で埋めるので、マイク音声はスピーカーに返しません。

`4096` サンプル単位でコールバックされても、そのまま VAD に入るわけではありません。リサンプラで `16 kHz` に落としてから、さらに `512` サンプル単位に切ります。

## リサンプラの仕様

`StreamingResampler` は `sourceRate / 16000` の比率で逐次変換する、線形補間ベースのシンプルなストリーミングリサンプラです。

- 入力は `Float32Array`
- 出力も `Float32Array`
- 内部に `pending` と `position` を持ち、コールバック境界をまたいでサンプルを保持
- サンプル数は毎回固定ではない

たとえば入力が `48 kHz` なら、`4096` サンプル入力ごとに概ね `1365` 前後のサンプルが出ますが、端数は `pending` と `position` に残ります。

## VAD の入力と出力

### 入力

VAD には `512 samples @ 16 kHz` のチャンクを渡します。1 チャンクは約 `32 ms` です。

モデル入力は次の 3 つです。

- `input`: `[1, 576]`
  `64` サンプルのコンテキスト + 今回の `512` サンプル
- `state`: `[2, 1, 128]`
  前回推論から引き継ぐ内部状態
- `sr`: scalar `16000`

### 出力

- 第 1 出力: 音声らしさの確率 `vadProb`
- 第 2 出力: 次回に引き継ぐ新しい `state`

`vadProb` 自体は連続値で、2 値化はアプリ側で行います。

- `vadProb >= vadThreshold` なら発話中
- `vadProb < vadThreshold` なら無音扱い

### 内部状態の扱い

- VAD の `state` と `context` は毎チャンク更新されます。
- ただし `5 秒` 以上リセットなしで走ると `maybeReset()` で初期化されます。
- さらに `resetCaptureState()` でも初期化されます。

そのため、VAD は完全な無限ストリーム状態を保ち続けるのではなく、「発話境界」や「長時間経過」で状態を切り直す実装です。

## バッファと状態の仕様

### `vadQueue`

- リサンプル済みサンプルの一次キュー
- 中身は JavaScript の配列
- `appendSamplesToQueue()` で 1 サンプルずつ push
- `drainVadQueue()` が `512` サンプル単位で先頭から取り出して処理

VAD 実行中は `vadBusy = true` になり、同時に複数の `drainVadQueue()` は走りません。

### `recentAudioChunks`

- Smart Turn へ渡すための入力バッファ
- VAD 状態に関係なく、すべての処理済みチャンクを追加
- 最大 `250 chunks = 8 秒` に制限

このバッファは VAD 判定より先に更新されるので、発話中の音声だけでなく、発話後の無音チャンクも含まれます。Smart Turn が実際に見るのは、このバッファを `8 秒固定長` に整形した音声です。

### `speechActive`

- `false` の間は待機状態
- `vadProb >= vadThreshold` を検出すると `true` になる
- Smart Turn 実行前の `resetCaptureState()` で `false` に戻る

この状態は「いま発話セグメントの中にいるか」を表していて、Smart Turn 実行のトリガー管理に使います。

### `pendingTurnTimeoutId`

- 発話終了候補を見つけたあとにセットする待機タイマー
- `vadProb < vadThreshold` になった最初のチャンクで開始
- 待機中に `vadProb >= vadThreshold` に戻ったらキャンセル
- 待機時間は `state.postSpeechMs` を使う

`Smart Turn delay` スライダーで変えているのは、この待機時間です。

## 発話開始と発話終了

### 発話開始

`speechActive = false` のときに `vadProb >= vadThreshold` になると:

1. そのチャンクはすでに `recentAudioChunks` に追加されている
2. `speechActive = true` になる
3. UI 状態は `Recording speech` に変わる

この実装には `preBuffer` はなく、発話直前をさかのぼって補う処理はありません。Smart Turn 入力の先頭は、`recentAudioChunks` に現在残っている履歴に依存します。

### 発話終了

`speechActive = true` 中に `vadProb < threshold` になっても、その瞬間にはまだ Smart Turn を実行しません。

1. 最初の無音チャンクも、すでに `recentAudioChunks` に追加されている
2. その時点で `Smart Turn delay` のタイマーを開始する
3. タイマー中も後続チャンクは通常どおり `recentAudioChunks` に追加される
4. その間に `vadProb >= threshold` に戻れば、タイマーを取り消して同じ発話を継続する
5. 設定した delay 時間だけ無音が続いたら、`recentAudioChunks` 全体を連結する
6. `prepareSmartTurnAudio()` で、その音声を `8 秒固定長` に整形する
7. `resetCaptureState()` で `speechActive` と VAD 状態を消す
8. Smart Turn を 1 回実行する
9. 判定結果が `Complete` なら、推論完了時に `recentAudioChunks` を空にする

ここで重要なのは、`VAD OFF` になった瞬間の音声だけでなく、`delay` の待機中に入った音声も Smart Turn 入力に含まれることです。

## Smart Turn delay スライダーの意味

- 初期値は `200 ms`
- 範囲は `0 ms` から `300 ms`
- 刻みは `10 ms`
- 実行中は変更できず、停止中に設定した値が次回の実行から使われる

たとえば `300 ms` に設定した場合は、`VAD ON -> OFF` になったあと約 `300 ms` 待ってから Smart Turn を実行します。その待機中に `recentAudioChunks` に追加された無音や後続音声も入力末尾に含まれます。

ただし音声処理は `512 samples` 単位、つまり約 `32 ms` 単位で進むので、境界は完全なサンプル単位ではなくチャンク粒度になります。

## Smart Turn の入力と出力

### 入力

Smart Turn に渡す音声は常に `Float32Array(128000)` です。元になる素材は、発話区間そのものではなく、`recentAudioChunks` にたまっている直近音声です。

- バッファ音声が `8 秒` を超える場合: `末尾 8 秒` を切り出す
- バッファ音声が `8 秒` 未満の場合: `先頭側を 0 埋め` して右寄せする

その後、特徴量抽出器で次を行います。

1. 平均 0、分散ベースのスケーリングで正規化
2. 両端を反射パディング
3. `n_fft = 400`, `hop_length = 160`, `80 mel`, `800 frames` で log-mel を作成
4. `globalMax - 8` でフロアをかけ、`(x + 4) / 4` でスケール

最終的な ONNX 入力テンソルは `input_features: [1, 80, 800]` です。

### 出力

Smart Turn モデルの第 1 出力を `probability` として読みます。これは「この発話がここで完結しているらしさ」の連続値です。

アプリ側では:

- `probability >= turnThreshold` なら `prediction = 1` (`Complete`)
- それ未満なら `prediction = 0` (`Incomplete`)

と判定します。

さらに:

- `Complete` の場合は、推論結果が返った直後に `recentAudioChunks` を空にする
- `Incomplete` の場合は、`recentAudioChunks` を保持し、次回の Smart Turn 入力に引き継ぐ

## 推論中に入ってくる音声の扱い

Smart Turn 推論中も `onaudioprocess` 自体は止まらないため、新しいマイク音声は引き続き `vadQueue` やリサンプラ内部にたまります。

ただし `drainVadQueue()` は `segmentProcessingPromise` がある間は待機するため、推論中は新しいチャンクに対する VAD 判定を進めません。推論完了後、たまっていたキューを先頭から再開します。

- `vadQueue` は保持される
- リサンプラの `pending` / `position` も保持される
- `recentAudioChunks` は `Complete` でのみ空になる
- 推論完了後、たまっていた音声が先頭から VAD に流れる

そのため、Smart Turn 判定中も録音自体は継続しますが、発話境界の判定処理は推論完了まで一時停止します。

## 状態遷移の見方

- `Listening`
  待機中。VAD だけ常時回す。
- `Recording speech`
  VAD が閾値を超えたあと、発話中または `VAD OFF` 後の delay 待機中。
- `Predicting`
  発話終端候補から設定した delay の静音継続が確認できたので Smart Turn を 1 回実行中。
- `Ready` / `Idle` / `Error`
  起動前後や異常時の UI 状態。

## 実装上の要点

- VAD は約 `32 ms` ごとの逐次判定
- Smart Turn は `VAD OFF` 直後ではなく、設定した delay の静音継続確認後に単発判定
- `Smart Turn delay` を変えると、Smart Turn 入力に含まれる末尾側の音声タイミングも変わる
- Smart Turn 入力は `recentAudioChunks` を元にした `8 秒固定長`
- 短い音声は `左ゼロ埋め`, 長い音声は `末尾だけ使用`
- `recentAudioChunks` は発話中以外のチャンクも含めて常時更新される
- `Incomplete` では Smart Turn 入力バッファを保持し、`Complete` でだけ空にする
- 実行中は delay スライダーを変更できない

## 参照箇所

- 定数定義: `RATE`, `CHUNK`, `INITIAL_POST_SPEECH_MS`, `MAX_DURATION_SECONDS`
- 音声入力開始: `start()`
- リサンプル: `StreamingResampler`
- VAD 実行: `SileroVAD.probability()`
- チャンク処理: `processChunk()`
- 発話終了後の推論: `finalizePendingTurn()`, `processSegment()`
- fixed-length 整形: `prepareSmartTurnAudio()`
- 特徴量抽出: `BuiltInWhisperFeatureExtractor.extract()`
