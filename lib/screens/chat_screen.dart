import 'package:flutter/material.dart';
import 'package:genkit/genkit.dart';
import '../models/message_model.dart';
import '../services/ai_engine.dart';
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

  bool _cloudReady = false;
  bool _localReady = false;
  bool _ragReady = false;

  late final AiEngine _engine;
  RagService? _ragService;

  PolicyMode _policy = PolicyMode.cloud;
  bool _ragEnabled = false;
  List<String> _lastRagSources = [];

  // Throttle setState during token streaming to avoid rebuilding on every token.
  DateTime _lastUiUpdate = DateTime.now();
  static const _uiUpdateInterval = Duration(milliseconds: 50);

  @override
  void initState() {
    super.initState();
    _engine = AiEngine();
    _initServices();
  }

  Future<void> _initServices() async {
    try {
      if (mounted)
        setState(() => _statusMessage = 'Downloading local model...');
      await _engine.initialize(
        onProgress: (p) {
          if (mounted) setState(() => _downloadProgress = p / 100);
        },
      );
    } catch (e) {
      debugPrint('AiEngine init failed: $e');
    }

    _cloudReady = _engine.cloudReady;
    _localReady = _engine.localReady;

    if (_localReady) {
      try {
        if (mounted) setState(() => _statusMessage = 'Setting up RAG...');
        final rag = RagService(
          ai: _engine.ai,
          embedderName: _engine.embedderName,
        );
        await rag.initialize(
          onStatus: (s) {
            if (mounted) setState(() => _statusMessage = s);
          },
        );
        _ragService = rag;
        _ragReady = true;
      } catch (e) {
        debugPrint('RAG init failed: $e');
      }
    }

    if (!mounted) return;
    final defaultPolicy = switch ((_cloudReady, _localReady)) {
      (true, _) => PolicyMode.cloud,
      (false, true) => PolicyMode.local,
      _ => PolicyMode.cloud,
    };
    final parts = [
      if (_cloudReady) 'cloud',
      if (_localReady) 'local',
      if (_ragReady) 'RAG',
    ];
    setState(() {
      _isInitializing = false;
      _policy = defaultPolicy;
      _statusMessage = parts.isEmpty
          ? 'No services available'
          : '${parts.join(', ')} ready';
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    _scrollController.dispose();
    _engine.dispose();
    _ragService?.dispose();
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
        final ragResult = await _ragService!.searchAndBuildContext(text);
        if (!mounted) return;
        if (ragResult.hasContext) {
          prompt = ragResult.augmentedPrompt;
          setState(() => _lastRagSources = ragResult.sources);
        }
      }

      final userMessage = Message(
        role: Role.user,
        content: [TextPart(text: prompt)],
      );

      final buffer = StringBuffer();
      _lastUiUpdate = DateTime.now();

      final stream = _engine.ai.generateStream(
        model: _engine.modelFor(_policy),
        messages: [userMessage],
      );
      await for (final chunk in stream) {
        buffer.write(chunk.text);
        final now = DateTime.now();
        if (now.difference(_lastUiUpdate) >= _uiUpdateInterval) {
          _lastUiUpdate = now;
          if (!mounted) return;
          setState(() {
            _messages.last = ChatMessage(
              text: buffer.toString(),
              isUser: false,
            );
          });
          _scrollToBottom();
        }
      }
      if (_policy == PolicyMode.budget || _policy == PolicyMode.cloud) {
        _engine.cloudCallsSpent++; // demo accounting for CostStrategy
      }

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
      if (mounted) {
        setState(() {
          _isGenerating = false;
          // Clean up empty placeholder if stream was interrupted before yielding.
          if (_messages.isNotEmpty &&
              !_messages.last.isUser &&
              _messages.last.text.isEmpty) {
            _messages.removeLast();
          }
        });
      }
    }
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
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            child: DropdownButton<PolicyMode>(
              value: _policy,
              isExpanded: true,
              items: [
                DropdownMenuItem(
                  value: PolicyMode.cloud,
                  enabled: _cloudReady,
                  child: const Text('Cloud'),
                ),
                DropdownMenuItem(
                  value: PolicyMode.local,
                  enabled: _localReady,
                  child: const Text('Local'),
                ),
                DropdownMenuItem(
                  value: PolicyMode.smart,
                  enabled: _cloudReady && _localReady,
                  child: const Text('Smart (image-aware)'),
                ),
                DropdownMenuItem(
                  value: PolicyMode.cascade,
                  enabled: _cloudReady && _localReady,
                  child: const Text('Cascade (escalate on quality)'),
                ),
                DropdownMenuItem(
                  value: PolicyMode.budget,
                  enabled: _cloudReady && _localReady,
                  child: const Text('Budget (cost-gated)'),
                ),
              ],
              onChanged: (m) {
                if (m != null) setState(() => _policy = m);
              },
            ),
          ),
          if (_lastRagSources.isNotEmpty)
            Container(
              width: double.infinity,
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
              color: Theme.of(context).colorScheme.tertiaryContainer,
              child: Row(
                children: [
                  Icon(
                    Icons.auto_awesome,
                    size: 16,
                    color: Theme.of(context).colorScheme.onTertiaryContainer,
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      'Sources: ${_lastRagSources.join(', ')}',
                      style: TextStyle(
                        fontSize: 12,
                        color: Theme.of(
                          context,
                        ).colorScheme.onTertiaryContainer,
                      ),
                    ),
                  ),
                ],
              ),
            ),
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
                    itemBuilder: (context, index) =>
                        MessageBubble(message: _messages[index]),
                  ),
          ),
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
                  Text('Generating...', style: TextStyle(color: Colors.grey)),
                ],
              ),
            ),
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
                          borderRadius: BorderRadius.all(Radius.circular(24)),
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
