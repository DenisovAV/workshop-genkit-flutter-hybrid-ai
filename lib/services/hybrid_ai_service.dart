import 'ai_service.dart';
import 'cloud_ai_service.dart';
import 'local_ai_service.dart';

enum AIStrategy { localFirst, cloudFirst, localOnly, cloudOnly }

class HybridAIService implements AIService {
  final LocalAIService local;
  final CloudAIService cloud;

  AIStrategy strategy = AIStrategy.localFirst;

  HybridAIService({required this.local, required this.cloud});

  @override
  Future<void> initialize() async {
    await cloud.initialize();
    await local.initialize();
  }

  @override
  Stream<String> generateResponseStream(String prompt) async* {
    switch (strategy) {
      case AIStrategy.localFirst:
        var yielded = false;
        try {
          await for (final chunk in local.generateResponseStream(prompt)) {
            yielded = true;
            yield chunk;
          }
        } catch (e) {
          if (yielded) rethrow; // don't silently switch mid-response
          yield* cloud.generateResponseStream(prompt);
        }
      case AIStrategy.cloudFirst:
        var yielded = false;
        try {
          await for (final chunk in cloud.generateResponseStream(prompt)) {
            yielded = true;
            yield chunk;
          }
        } catch (e) {
          if (yielded) rethrow;
          yield* local.generateResponseStream(prompt);
        }
      case AIStrategy.localOnly:
        yield* local.generateResponseStream(prompt);
      case AIStrategy.cloudOnly:
        yield* cloud.generateResponseStream(prompt);
    }
  }

  @override
  Future<void> dispose() async {
    await local.dispose();
    await cloud.dispose();
  }
}
