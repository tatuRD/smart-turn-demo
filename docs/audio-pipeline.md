# Audio / VAD / Smart Turn 仕様メモ

`docs/app.js` の実装をもとに、このデモで音声がどのように取り込まれ、VAD と smart-turn に渡され、各種バッファがどう扱われるかを整理したメモです。

## 全体像

1. ブラウザのマイク入力を 1ch で取得する。
2. `AudioContext` の入力サンプルレートから `16 kHz` に逐次リサンプルする。
3. リサンプル済みサンプルを `vadQueue` に積み、`512 samples` ごとに VAD を実行する。
4. VAD が発話開始を検出したら、直前の `preBuffer` を含めて `segment` に取り込み続ける。
5. 同時に、連続音声の直近 `8 秒` は常に別バッファに保持し続ける。
6. VAD が無音に戻った最初のチャンクで発話を閉じ、その時点の「直近 8 秒」を smart-turn に渡す。
7. smart-turn の確率値をしきい値で 2 値化し、`Complete` / `Incomplete` を決める。

この実装では smart-turn はストリーミング推論ではなく、`発話終了時に 1 回だけ` 実行されます。

## 主要パラメータ

| 項目 | 値 | 意味 |
| --- | --- | --- |
| VAD / smart-turn 用レート | `16000 Hz` | すべてのモデル入力は 16 kHz 前提 |
| VAD チャンク | `512 samples` | `32 ms` ごとに VAD を回す |
| VAD 先読み保持 | `200 ms` | 実際には `7 chunks = 224 ms` 保持 |
| smart-turn 最大長 | `8 s` | `128000 samples` |
| smart-turn フレーム数 | `800 frames` | `hop_length = 160` に対応 |
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

`4096` サンプル単位でコールバックされても、そのまま VAD に入るわけではありません。次のリサンプラで `16 kHz` に落としてから、さらに `512` サンプル単位に切ります。

## リサンプラの仕様

`StreamingResampler` は `sourceRate / 16000` の比率で逐次変換する、線形補間ベースのシンプルなストリーミングリサンプラです。

- 入力は `Float32Array`
- 出力も `Float32Array`
- 内部に `pending` と `position` を持ち、コールバック境界をまたいでサンプルを保持
- サンプル数は毎回固定ではない

たとえば入力が `48 kHz` なら、`4096` サンプル入力ごとに概ね `1365` 前後のサンプルが出ますが、端数は `pending` と `position` に残ります。

## VAD の入力と出力

### 入力

VAD には `512 samples @ 16 kHz` のチャンクを渡します。1 チャンクは `32 ms` です。

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

## 発話区間の切り出しとバッファの仕様

### `vadQueue`

- リサンプル済みサンプルの一次キュー
- 中身は JavaScript の配列
- `appendSamplesToQueue()` で 1 サンプルずつ push
- `drainVadQueue()` が `512` サンプル単位で先頭から取り出して処理

VAD 実行中は `vadBusy = true` になり、同時に複数の `drainVadQueue()` は走りません。

### `preBuffer`

- 発話開始前の履歴保持用
- 最大 `preChunks = 7` 個
- `7 * 512 = 3584 samples`
- `3584 / 16000 = 224 ms`

コード上の設定は `200 ms` ですが、実際の保持量はチャンク境界に丸められて `224 ms` です。

まだ発話していない状態では、すべてのチャンクが一度 `preBuffer` に入ります。

### `segment`

- 発話区間の長さや境界を追うための発話候補
- 発話開始時に `segment = preBuffer.slice()` で初期化
- 以後、発話中チャンクを順次追加
- 最大 `250 chunks = 8 秒` に制限

上限を超えたら先頭から捨てるので、`8 秒を超える発話は末尾 8 秒だけ` が残ります。

### `recentAudioChunks`

- smart-turn へ渡すための連続ローリング音声
- VAD 状態に関係なく、すべての処理済みチャンクを追加
- 最大 `250 chunks = 8 秒` に制限

こちらは `segment` と違って、「今が発話中かどうか」に関係なく動き続けます。smart-turn が実際に見るのはこのローリング 8 秒です。

### 発話開始

まだ `speechActive = false` のときに `vadProb >= threshold` になると:

1. そのチャンクはすでに `preBuffer` に入っている
2. `segment = preBuffer.slice()` で直前の履歴ごとコピーされる
3. `speechActive = true` になる

つまり smart-turn 用の発話先頭には、VAD が立ち上がる直前の約 `224 ms` ぶんも含まれます。

### 発話終了

`speechActive = true` 中に `vadProb < threshold` になった最初のチャンクで:

1. その無音チャンクもいったん `segment` に追加される
2. `segment` 全体を連結して発話長を求める
3. `recentAudioChunks` 全体を連結し、その時点の直近音声を得る
4. `prepareSmartTurnAudio()` で、その直近音声を `8 秒固定長` に整形する
5. `resetCaptureState()` で `preBuffer` / `segment` / VAD 状態を消す
6. smart-turn を 1 回実行する

ここで重要なのは、`発話終了を決めた無音チャンクも segment` と `recentAudioChunks` の両方に含まれることです。

## smart-turn の入力と出力

### 入力

smart-turn に渡す音声は常に `Float32Array(128000)` です。元になる素材は、発話区間そのものではなく、`発話終了時点での直近 8 秒のローリング音声` です。

- 直近音声が `8 秒` を超える場合: `末尾 8 秒` を切り出す
- 直近音声が `8 秒` 未満の場合: `先頭側を 0 埋め` して右寄せする

その後、特徴量抽出器で次を行います。

1. 平均 0、分散ベースのスケーリングで正規化
2. 両端を反射パディング
3. `n_fft = 400`, `hop_length = 160`, `80 mel`, `800 frames` で log-mel を作成
4. `globalMax - 8` でフロアをかけ、`(x + 4) / 4` でスケール

最終的な ONNX 入力テンソルは:

- `input_features`: `[1, 80, 800]`

です。

### 出力

smart-turn モデルの第 1 出力を `probability` として読みます。これは「この発話がここで完結しているらしさ」の連続値です。

アプリ側では:

- `probability >= turnThreshold` なら `prediction = 1` (`Complete`)
- それ未満なら `prediction = 0` (`Incomplete`)

と判定します。

## 推論中に入ってくる音声の扱い

smart-turn 推論中も `onaudioprocess` 自体は止まらないため、新しいマイク音声は引き続き `vadQueue` やリサンプラ内部にたまります。

`processSegment()` に入る前の `processChunk()` で `resetCaptureState()` はすでに実行されているため、推論中に入ってきた音声は「前の発話の続き」ではなく「次の発話候補」として扱われます。

- `vadQueue` は保持される
- リサンプラの `pending` / `position` も保持される
- `recentAudioChunks` も保持される
- 推論完了後、たまっていた音声が先頭から VAD に流れる
- その結果、後続音声は次の `preBuffer` / `segment` に取り込まれつつ、ローリング 8 秒にも継続して残る

つまり、`Complete` / `Incomplete` に関係なく、推論中に入ってきた後続音声は捨てません。実装上は「smart-turn 判定中にも次ターンの録音素材を溜め続ける」挙動です。

## 状態遷移の見方

- `Listening`
  待機中。VAD だけ常時回す。
- `Recording speech`
  VAD が閾値を超え、`segment` に発話を貯めている状態。
- `Predicting`
  発話終端が来たので smart-turn を 1 回実行中。
- `Ready` / `Idle` / `Error`
  起動前後や異常時の UI 状態。

## 実装上の要点

- VAD は `32 ms` ごとの逐次判定
- smart-turn は `発話終端ごとの単発判定`
- smart-turn 入力は常に `直近 8 秒` を元にした `8 秒固定長`
- 短い音声は `左ゼロ埋め`, 長い音声は `末尾だけ使用`
- `preBuffer` により、発話先頭は約 `224 ms` さかのぼって保持
- 発話末尾には、終端判定に使った最初の無音チャンクも含まれる
- `segment` は発話境界の管理用で、smart-turn の実入力は `recentAudioChunks` から作る
- smart-turn 推論中に入ってきた後続音声も保持し、次の発話候補として処理する

## 参照箇所

- 定数定義: `RATE`, `CHUNK`, `PRE_SPEECH_MS`, `MAX_DURATION_SECONDS`
- 音声入力開始: `start()`
- リサンプル: `StreamingResampler`
- VAD 実行: `SileroVAD.probability()`
- チャンク処理: `processChunk()`
- 発話終了後の推論: `processSegment()`
- fixed-length 整形: `prepareSmartTurnAudio()`
- 特徴量抽出: `BuiltInWhisperFeatureExtractor.extract()`
