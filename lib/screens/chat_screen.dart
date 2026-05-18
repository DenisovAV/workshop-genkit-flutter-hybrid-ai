import 'package:flutter/material.dart';
import '../models/message_model.dart';
import '../services/cloud_ai_service.dart';
import '../services/hybrid_ai_service.dart';
import '../services/local_ai_service.dart';
import '../services/rag_service.dart';
import '../widgets/message_bubble.dart';

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final _controller = TextEditingController();
  final _scrollController = ScrollController();
  final _messages = <ChatMessage>[];

  bool _isGenerating = false;
  bool _isInitializing = true;
  double _downloadProgress = 0;
  String _statusMessage = 'Initializing...';

  // Per-service readiness flags — used to enable/disable UI segments.
  bool _cloudReady = false;
  bool _localReady = false;
  bool _ragReady = false;

  // Services are assigned at the start of _initServices before any await,
  // so late fields are always valid by the time the UI is interactive.
  late final CloudAIService _cloudService;
  late final LocalAIService _localService;
  late final HybridAIService _hybridService;
  late final RagService _ragService;

  AIStrategy _strategy = AIStrategy.cloudOnly;
  bool _ragEnabled = false;
  List<String> _lastRagSources = [];

  // Throttle setState during token streaming to avoid rebuilding on every token.
  DateTime _lastUiUpdate = DateTime.now();
  static const _uiUpdateInterval = Duration(milliseconds: 50);

  @override
  void initState() {
    super.initState();
    _cloudService = CloudAIService();
    _localService = LocalAIService();
    _hybridService = HybridAIService(local: _localService, cloud: _cloudService);
    _ragService = RagService();
    _initServices();
  }

  Future<void> _initServices() async {
    // Initialize each service independently so a failure in one
    // does not prevent others from starting.
    try {
      if (mounted) setState(() => _statusMessage = 'Connecting to cloud AI...');
      await _cloudService.initialize();
      _cloudReady = true;
    } catch (e) {
      debugPrint('Cloud service init failed: $e');
    }

    try {
      if (mounted) setState(() => _statusMessage = 'Downloading local model...');
      await _localService.initialize(
        onProgress: (progress) {
          if (mounted) setState(() => _downloadProgress = progress);
        },
      );
      _localReady = true;
    } catch (e) {
      debugPrint('Local service init failed: $e');
    }

    try {
      if (mounted) setState(() => _statusMessage = 'Setting up RAG...');
      await _ragService.initialize(
        onStatus: (status) {
          if (mounted) setState(() => _statusMessage = status);
        },
      );
      _ragReady = true;
    } catch (e) {
      debugPrint('RAG service init failed: $e');
    }

    if (!mounted) return;

    final defaultStrategy = switch ((_cloudReady, _localReady)) {
      (true, true) => AIStrategy.localFirst,
      (false, true) => AIStrategy.localOnly,
      (true, false) => AIStrategy.cloudOnly,
      _ => AIStrategy.cloudOnly,
    };

    _hybridService.strategy = defaultStrategy;

    final parts = [
      if (_cloudReady) 'cloud',
      if (_localReady) 'local',
      if (_ragReady) 'RAG',
    ];
    final status = parts.isEmpty
        ? 'No services available'
        : '${parts.join(', ')} ready';

    setState(() {
      _isInitializing = false;
      _strategy = defaultStrategy;
      _statusMessage = status;
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    _scrollController.dispose();
    _hybridService.dispose();
    _ragService.dispose();
    super.dispose();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  Future<void> _sendMessage() async {
    final text = _controller.text.trim();
    if (text.isEmpty || _isGenerating) return;

    _controller.clear();

    setState(() {
      _messages.add(ChatMessage(text: text, isUser: true));
      _messages.add(ChatMessage(text: '', isUser: false));
      _isGenerating = true;
      _lastRagSources = [];
    });
    _scrollToBottom();

    try {
      String prompt = text;

      if (_ragEnabled && _ragReady) {
        final ragResult = await _ragService.searchAndBuildContext(text);
        if (!mounted) return;
        if (ragResult.hasContext) {
          prompt = ragResult.augmentedPrompt;
          setState(() => _lastRagSources = ragResult.sources);
        }
      }

      final buffer = StringBuffer();
      _lastUiUpdate = DateTime.now();

      await for (final chunk in _hybridService.generateResponseStream(prompt)) {
        buffer.write(chunk);

        final now = DateTime.now();
        if (now.difference(_lastUiUpdate) >= _uiUpdateInterval) {
          _lastUiUpdate = now;
          if (!mounted) return;
          setState(() {
            _messages.last = ChatMessage(text: buffer.toString(), isUser: false);
          });
          _scrollToBottom();
        }
      }

      // Render the final accumulated text after the stream ends.
      if (!mounted) return;
      setState(() {
        _messages.last = ChatMessage(text: buffer.toString(), isUser: false);
      });
      _scrollToBottom();
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _messages.last = ChatMessage(text: 'Error: $e', isUser: false);
      });
    } finally {
      if (mounted) setState(() => _isGenerating = false);
    }
  }

  void _onStrategyChanged(AIStrategy strategy) {
    setState(() {
      _strategy = strategy;
      _hybridService.strategy = strategy;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('AI Chat'),
        centerTitle: true,
        actions: [
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text('RAG', style: TextStyle(fontSize: 12)),
              Switch(
                value: _ragEnabled,
                onChanged: _ragReady
                    ? (value) => setState(() => _ragEnabled = value)
                    : null,
              ),
            ],
          ),
        ],
      ),
      body: Column(
        children: [
          // Strategy picker — segments disabled when the service isn't ready.
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            child: SegmentedButton<AIStrategy>(
              segments: [
                ButtonSegment(
                  value: AIStrategy.cloudOnly,
                  label: const Text('Cloud'),
                  icon: const Icon(Icons.cloud),
                  enabled: _cloudReady,
                ),
                ButtonSegment(
                  value: AIStrategy.localOnly,
                  label: const Text('Local'),
                  icon: const Icon(Icons.phone_android),
                  enabled: _localReady,
                ),
                ButtonSegment(
                  value: AIStrategy.localFirst,
                  label: const Text('Local+Cloud'),
                  icon: const Icon(Icons.download_for_offline),
                  enabled: _cloudReady && _localReady,
                ),
                ButtonSegment(
                  value: AIStrategy.cloudFirst,
                  label: const Text('Cloud+Local'),
                  icon: const Icon(Icons.cloud_sync),
                  enabled: _cloudReady && _localReady,
                ),
              ],
              selected: {_strategy},
              onSelectionChanged: (selected) {
                _onStrategyChanged(selected.first);
              },
            ),
          ),

          // RAG sources banner
          if (_lastRagSources.isNotEmpty)
            Container(
              width: double.infinity,
              padding:
                  const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
              color: Theme.of(context).colorScheme.tertiaryContainer,
              child: Row(
                children: [
                  Icon(
                    Icons.auto_awesome,
                    size: 16,
                    color:
                        Theme.of(context).colorScheme.onTertiaryContainer,
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      'Sources: ${_lastRagSources.join(', ')}',
                      style: TextStyle(
                        fontSize: 12,
                        color: Theme.of(context)
                            .colorScheme
                            .onTertiaryContainer,
                      ),
                    ),
                  ),
                ],
              ),
            ),

          // Initialization progress
          if (_isInitializing)
            Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                children: [
                  Text(_statusMessage),
                  const SizedBox(height: 8),
                  LinearProgressIndicator(
                    value: _downloadProgress > 0 ? _downloadProgress : null,
                  ),
                ],
              ),
            ),

          // Messages list
          Expanded(
            child: _messages.isEmpty
                ? const Center(
                    child: Text(
                      'Send a message to start chatting',
                      style: TextStyle(color: Colors.grey),
                    ),
                  )
                : ListView.builder(
                    controller: _scrollController,
                    padding: const EdgeInsets.only(top: 8, bottom: 8),
                    itemCount: _messages.length,
                    itemBuilder: (context, index) {
                      return MessageBubble(message: _messages[index]);
                    },
                  ),
          ),

          // Generating indicator
          if (_isGenerating)
            const Padding(
              padding: EdgeInsets.symmetric(horizontal: 16, vertical: 4),
              child: Row(
                children: [
                  SizedBox(
                    width: 16,
                    height: 16,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  ),
                  SizedBox(width: 8),
                  Text(
                    'Generating...',
                    style: TextStyle(color: Colors.grey),
                  ),
                ],
              ),
            ),

          // Input field
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: Theme.of(context).colorScheme.surface,
              border: Border(
                top: BorderSide(
                  color: Theme.of(context).colorScheme.outlineVariant,
                ),
              ),
            ),
            child: SafeArea(
              child: Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _controller,
                      decoration: const InputDecoration(
                        hintText: 'Type a message...',
                        border: OutlineInputBorder(
                          borderRadius:
                              BorderRadius.all(Radius.circular(24)),
                        ),
                        contentPadding: EdgeInsets.symmetric(
                          horizontal: 16,
                          vertical: 10,
                        ),
                      ),
                      textInputAction: TextInputAction.send,
                      onSubmitted: (_) => _sendMessage(),
                    ),
                  ),
                  const SizedBox(width: 8),
                  IconButton.filled(
                    onPressed: (_isGenerating || _isInitializing)
                        ? null
                        : _sendMessage,
                    icon: const Icon(Icons.send),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
