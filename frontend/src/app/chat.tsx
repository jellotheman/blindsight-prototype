import {
  checkCapturedView,
  createQuestion,
  pollQuestion,
  type QuestionResource,
} from '@/services/api';
import { useLocalSearchParams, useRouter } from 'expo-router';
import * as Speech from 'expo-speech';
import { useEffect, useRef, useState } from 'react';
import {
  FlatList,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

type Message = { id: number; role: 'user' | 'assistant'; text: string };

// The card had no grounds for an answer. The offer is spoken before any captured-view
// check is requested, and it warns about the extra wait the user is accepting.
const CONSENT_OFFER =
  "I couldn't answer that from the scene card. Shall I check the captured view again? " +
  'This may take several seconds.';
// A second miss is a plain abstention. It is never turned into a confident negative claim.
const ABSTENTION = "I couldn't tell from the capture.";

export default function ChatScreen() {
  const router = useRouter();
  const insets = useSafeAreaInsets();
  const params = useLocalSearchParams<{ sceneSessionId: string }>();
  const sceneSessionId = typeof params.sceneSessionId === 'string' ? params.sceneSessionId : '';

  const nextId = useRef(1);
  const busyRef = useRef(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [consent, setConsent] = useState<{ questionId: string } | null>(null);

  const appendMessage = (role: Message['role'], text: string) => {
    setMessages((current) => [
      ...current,
      { id: nextId.current++, role, text },
    ]);
  };

  // One place where a settled question becomes a message, shared by the card answer and
  // the captured-view answer. A miss is spoken as a miss; it never becomes a confident negative.
  const speakOutcome = (settled: QuestionResource) => {
    if (settled.status === 'answered') {
      appendMessage('assistant', settled.answer ?? '');
      Speech.speak(settled.answer ?? '', { rate: 0.9 });
    } else if (settled.status === 'needs_clip_consent') {
      setConsent({ questionId: settled.question_id });
      appendMessage('assistant', CONSENT_OFFER);
      Speech.speak(CONSENT_OFFER, { rate: 0.9 });
    } else if (settled.status === 'unanswerable') {
      appendMessage('assistant', ABSTENTION);
      Speech.speak(ABSTENTION, { rate: 0.9 });
    } else {
      appendMessage('assistant', settled.failure?.message ?? 'Something went wrong.');
    }
  };

  const ask = async (question: string) => {
    if (busyRef.current || !sceneSessionId) {
      return;
    }
    busyRef.current = true;
    setBusy(true);
    setConsent(null);
    appendMessage('user', question);
    setInput('');
    try {
      const created = await createQuestion(sceneSessionId, question);
      const settled = await pollQuestion(sceneSessionId, created.question_id);
      speakOutcome(settled);
    } catch (err: unknown) {
      appendMessage('assistant', err instanceof Error ? err.message : 'Something went wrong.');
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  };

  const agreeToClipCheck = async () => {
    if (busyRef.current || !consent || !sceneSessionId) {
      return;
    }
    busyRef.current = true;
    setBusy(true);
    const { questionId } = consent;
    setConsent(null);
    try {
      await checkCapturedView(sceneSessionId, questionId);
      const settled = await pollQuestion(sceneSessionId, questionId);
      speakOutcome(settled);
    } catch (err: unknown) {
      appendMessage('assistant', err instanceof Error ? err.message : 'Something went wrong.');
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  };

  const refuseClipCheck = () => {
    if (busyRef.current || !consent) {
      return;
    }
    setConsent(null);
    // Declining leaves the question unanswered. The acknowledgement confirms the control was
    // heard without claiming anything about the scene.
    Speech.speak('All right.', { rate: 0.9 });
  };

  // Leaving the chat does not end the scene session: the capture screen that owns it ends
  // it on a new capture or when leaving. Deleting here would strand the parent's result
  // screen pointing at a session that no longer exists.
  const done = () => {
    Speech.stop();
    router.back();
  };

  useEffect(() => {
    if (!sceneSessionId && messages.length === 0) {
      appendMessage(
        'assistant',
        'Capture a scene first, then ask about it here.',
      );
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <View
        style={[
          styles.header,
          { height: insets.top + 48, paddingTop: insets.top + 6 },
        ]}
      >
        <Pressable
          onPress={() => {
            Speech.stop();
            router.back();
          }}
          accessibilityRole="button"
          accessibilityLabel="Back to the scene"
        >
          <Text style={styles.back}>‹ Back</Text>
        </Pressable>

        <Text style={styles.headerTitle}>ASK ABOUT WHAT YOU LOOKED AT</Text>

        <Pressable
          onPress={done}
          disabled={busy}
          accessibilityRole="button"
          accessibilityLabel="Done with this conversation"
        >
          <Text style={styles.done}>DONE</Text>
        </Pressable>
      </View>

      <FlatList
        style={styles.messages}
        contentContainerStyle={styles.messageList}
        data={messages}
        keyExtractor={(item) => String(item.id)}
        renderItem={({ item }) => (
          <View
            style={
              item.role === 'user' ? styles.userBubble : styles.assistantBubble
            }
          >
            <Text
              style={
                item.role === 'user' ? styles.userText : styles.assistantText
              }
            >
              {item.text}
            </Text>
          </View>
        )}
      />

      {consent ? (
        <View style={styles.consentRow}>
          <Pressable
            style={[styles.consentButton, styles.consentYes]}
            onPress={agreeToClipCheck}
            disabled={busy}
            accessibilityRole="button"
            accessibilityLabel="Yes, check the captured view"
          >
            <Text style={styles.consentYesText}>YES</Text>
          </Pressable>
          <Pressable
            style={[styles.consentButton, styles.consentNo]}
            onPress={refuseClipCheck}
            disabled={busy}
            accessibilityRole="button"
            accessibilityLabel="No, skip the captured view"
          >
            <Text style={styles.consentNoText}>NO</Text>
          </Pressable>
        </View>
      ) : null}

      <View style={styles.inputRow}>
        <TextInput
          style={styles.input}
          value={input}
          onChangeText={setInput}
          placeholder="Ask about what you looked at…"
          placeholderTextColor="#666970"
          editable={!busy && Boolean(sceneSessionId)}
          multiline
          accessibilityLabel="Your question"
        />
        <Pressable
          style={[styles.sendButton, (!input.trim() || busy) && styles.sendDisabled]}
          onPress={() => ask(input.trim())}
          disabled={!input.trim() || busy}
          accessibilityRole="button"
          accessibilityLabel="Send question"
        >
          <Text style={styles.sendText}>ASK</Text>
        </Pressable>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#08090B',
  },
  header: {
    paddingHorizontal: 20,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#08090B',
  },
  back: {
    color: '#FFFFFF',
    fontSize: 17,
    fontWeight: '700',
  },
  headerTitle: {
    color: '#FFFFFF',
    fontSize: 13,
    fontWeight: '800',
    letterSpacing: 1,
  },
  done: {
    color: '#08C8F8',
    fontSize: 13,
    fontWeight: '800',
  },
  messages: {
    flex: 1,
  },
  messageList: {
    padding: 20,
    gap: 12,
  },
  userBubble: {
    alignSelf: 'flex-end',
    maxWidth: '85%',
    backgroundColor: '#08C8F8',
    borderRadius: 16,
    borderBottomRightRadius: 4,
    paddingHorizontal: 14,
    paddingVertical: 10,
  },
  userText: {
    color: '#000000',
    fontSize: 15,
    lineHeight: 21,
  },
  assistantBubble: {
    alignSelf: 'flex-start',
    maxWidth: '85%',
    backgroundColor: '#191A1F',
    borderRadius: 16,
    borderBottomLeftRadius: 4,
    borderWidth: 1,
    borderColor: '#2C2E35',
    paddingHorizontal: 14,
    paddingVertical: 10,
  },
  assistantText: {
    color: '#FFFFFF',
    fontSize: 15,
    lineHeight: 21,
  },
  consentRow: {
    flexDirection: 'row',
    gap: 12,
    paddingHorizontal: 20,
    paddingVertical: 8,
  },
  consentButton: {
    flex: 1,
    padding: 14,
    borderRadius: 13,
    alignItems: 'center',
  },
  consentYes: {
    backgroundColor: '#08C8F8',
  },
  consentNo: {
    backgroundColor: '#24262D',
    borderWidth: 1,
    borderColor: '#3A3D45',
  },
  consentYesText: {
    color: '#000000',
    fontSize: 13,
    fontWeight: '800',
  },
  consentNoText: {
    color: '#FFFFFF',
    fontSize: 13,
    fontWeight: '800',
  },
  inputRow: {
    flexDirection: 'row',
    gap: 10,
    padding: 20,
    paddingBottom: 30,
  },
  input: {
    flex: 1,
    backgroundColor: '#191A1F',
    borderWidth: 1,
    borderColor: '#2C2E35',
    borderRadius: 14,
    color: '#FFFFFF',
    fontSize: 15,
    paddingHorizontal: 14,
    paddingVertical: 10,
    maxHeight: 100,
  },
  sendButton: {
    backgroundColor: '#08C8F8',
    borderRadius: 14,
    paddingHorizontal: 18,
    justifyContent: 'center',
    alignItems: 'center',
  },
  sendDisabled: {
    opacity: 0.4,
  },
  sendText: {
    color: '#000000',
    fontSize: 13,
    fontWeight: '800',
  },
});
