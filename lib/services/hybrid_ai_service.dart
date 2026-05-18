import 'cloud_ai_service.dart';
import 'local_ai_service.dart';

enum AIStrategy { localFirst, cloudFirst, localOnly, cloudOnly }

/// Routes generation requests between cloud and local AI services.
///
/// Expects both [local] and [cloud] to be independently initialized before
/// calling [generateResponseStream]. The [strategy] must only be changed
/// between requests, not during an active stream.
class HybridAIService {
  final LocalAIService local;
  final CloudAIService cloud;

  AIStrategy strategy = AIStrategy.localFirst;

  HybridAIService({required this.local, required this.cloud});

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

  Future<void> dispose() async {
    await local.dispose();
    await cloud.dispose();
  }
}
