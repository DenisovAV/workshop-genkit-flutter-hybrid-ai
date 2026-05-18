# Hybrid AI in Flutter: From Cloud to On-Device with Genkit Dart

A workshop demonstrating hybrid AI in Flutter using the [Genkit Dart](https://pub.dev/packages/genkit) framework: streaming responses from Gemini via `genkit_google_genai`, on-device inference via `genkit_flutter_gemma`, automatic cloud/local fallback, and a RAG pipeline using on-device embeddings.

## Workshop

Step-by-step instructions: [codelab/index.md](codelab/index.md)

| Branch | What's added |
|--------|-------------|
| `step-00-starter` | Chat UI with echo responses |
| `step-01-cloud-ai` | Cloud chat via `genkit_google_genai` (Gemini 2.5 Flash) |
| `step-02-local-ai` | On-device inference via `genkit_flutter_gemma` (Gemma 3 1B) |
| `step-03-hybrid` | Automatic fallback between cloud and local |
| `step-04-embeddings` | Text embeddings with `genkit_flutter_gemma` embedder |
| `step-05-rag` | RAG pipeline with in-memory tourist guide vector store |
| `main` | Complete project |

## Requirements

- Flutter 3.27+
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
genkit: ^0.13.0
genkit_google_genai: ^0.2.7
genkit_flutter_gemma: ^0.3.1
flutter_gemma: ^0.15.1
```

## Architecture

Both cloud and on-device inference use the same Genkit API:

```dart
// Cloud (Gemini 2.5 Flash)
ai.generateStream(model: googleAI.gemini('gemini-2.5-flash'), prompt: prompt)

// On-device (Gemma 3 1B)
ai.generateStream(model: flutterGemma.model('gemma-3-1b-it'), prompt: prompt)

// Embeddings (EmbeddingGemma 300M)
ai.embed(embedder: flutterGemma.embedder('embedding-gemma-300m'), document: ...)
```

The routing logic (`HybridAIService`) is plain Dart — Genkit lives inside each service, not in the router.
