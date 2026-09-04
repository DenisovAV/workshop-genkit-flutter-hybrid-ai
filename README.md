# Hybrid AI in Flutter: From Cloud to On-Device with Genkit Dart

A workshop showing hybrid AI in Flutter with the [Genkit Dart](https://pub.dev/packages/genkit) framework: streaming from Gemini via `genkit_google_genai`, on-device inference via `genkit_flutter_gemma` (LiteRT-LM), cloud↔on-device routing via [`genkit_hybrid`](https://pub.dev/packages/genkit_hybrid) (capability / cost / cascade), multimodal image input, and a RAG pipeline over on-device embeddings.

## Workshop

**📖 Full step-by-step codelab: https://fluttergemma.dev/codelabs/hybrid-ai-flutter-genkit/**

This repo holds the **code**. Check out a per-step branch to follow along, or to see the finished state of each step:

| Branch | What's added |
|--------|-------------|
| `step-00-starter` | Chat UI with echo responses |
| `step-01-cloud-ai` | Cloud chat via `genkit_google_genai` (Gemini 3.7 Flash) |
| `step-02-local-ai` | On-device inference via `genkit_flutter_gemma` (Gemma 3 1B, LiteRT-LM) |
| `step-03-hybrid` | `AiEngine` — one Genkit, cloud + local routed by `genkit_hybrid` |
| `step-04-smart-routing` | Smart routing + image (multimodal) input |
| `step-05-embeddings` | On-device embeddings + RAG over a local tourist guide |
| `main` | Complete project |

## Requirements

- Flutter 3.47.2 (latest stable)
- A GEMINI_API_KEY from [aistudio.google.com](https://aistudio.google.com)
- A HuggingFace account (for model downloads)
- Android device/emulator, iOS simulator, or macOS
- ~1 GB free disk space

## Quick Start

```bash
git clone https://github.com/DenisovAV/workshop-genkit-flutter-hybrid-ai.git
cd workshop-genkit-flutter-hybrid-ai
flutter pub get
flutter run \
  --dart-define=GEMINI_API_KEY=your_key \
  --dart-define=HF_TOKEN=hf_xxx
```

## Key Dependencies

```yaml
genkit: ^0.15.1
genkit_google_genai: ^0.2.12
genkit_flutter_gemma: ^0.5.0
flutter_gemma: ^1.7.0
flutter_gemma_litertlm: ^1.6.1   # LiteRT-LM engine (flutter_gemma 1.x ships none by default)
genkit_hybrid: ^0.2.0
```

## Architecture

One `Genkit` instance with both plugins (`googleAI` + `GenkitFlutterGemmaPlugin`), wrapped in an `AiEngine`. Each `PolicyMode` (cloud / local / smart / cascade / budget) is composed by `genkit_hybrid` into an ordinary Genkit `Model`; the chat screen always calls the same `ai.generateStream(...)` and only the resolved model changes with the policy:

```dart
// Cloud (Gemini 3.7 Flash)
ai.generateStream(model: googleAI.gemini('gemini-3.7-flash'), prompt: prompt)

// On-device (Gemma 3 1B via LiteRT-LM)
ai.generateStream(model: flutterGemma.model('gemma-3-1b-it'), prompt: prompt)

// Embeddings (EmbeddingGemma 300M)
ai.embed(embedder: flutterGemma.embedder('embedding-gemma-300m'), document: ...)
```

See the [codelab](https://fluttergemma.dev/codelabs/hybrid-ai-flutter-genkit/) for the full walkthrough.
