import { currentCredentials, saveCredentials } from '@/services/credentials';
import { useRouter } from 'expo-router';
import { useState } from 'react';
import { Pressable, StyleSheet, Text, TextInput, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

export default function SettingsScreen() {
  const router = useRouter();
  const insets = useSafeAreaInsets();
  const initial = currentCredentials();
  const [url, setUrl] = useState(initial.url);
  const [key, setKey] = useState(initial.key);
  const [status, setStatus] = useState('');

  const save = async () => {
    try {
      await saveCredentials(url, key);
      setStatus('Saved on this device.');
    } catch (err) {
      setStatus(err instanceof Error ? err.message : 'Could not save.');
    }
  };

  return (
    <View style={[styles.container, { paddingTop: insets.top + 20 }]}>
      <Pressable
        onPress={() => router.back()}
        accessibilityRole="button"
        accessibilityLabel="Go back"
      >
        <Text style={styles.back}>‹ Back</Text>
      </Pressable>

      <Text style={styles.title}>Settings</Text>

      <View style={styles.option}>
        <Text style={styles.optionTitle}>Server address</Text>
        <Text style={styles.optionText}>
          The BlindSight server address — the tunnel URL printed by the local
          dev tool, or a deployed URL. No trailing slash.
        </Text>
        <TextInput
          style={styles.input}
          value={url}
          onChangeText={setUrl}
          autoCapitalize="none"
          autoCorrect={false}
          keyboardType="url"
          placeholder="https://….trycloudflare.com"
          placeholderTextColor="#666970"
          accessibilityLabel="API URL"
        />
      </View>

      <View style={styles.option}>
        <Text style={styles.optionTitle}>API key</Text>
        <Text style={styles.optionText}>
          The shared X-API-Key. Stored in the device keychain, not in the public repo.
        </Text>
        <TextInput
          style={styles.input}
          value={key}
          onChangeText={setKey}
          autoCapitalize="none"
          autoCorrect={false}
          secureTextEntry
          placeholder="shared key"
          placeholderTextColor="#666970"
          accessibilityLabel="API key"
        />
      </View>

      <Pressable
        style={styles.saveButton}
        onPress={save}
        accessibilityRole="button"
        accessibilityLabel="Save API settings"
      >
        <Text style={styles.saveText}>SAVE</Text>
      </Pressable>

      {status ? <Text style={styles.status}>{status}</Text> : null}

      <View style={styles.option}>
        <Text style={styles.optionTitle}>Accessibility</Text>
        <Text style={styles.optionText}>
          Large text, high contrast, reduced motion and haptic feedback will be
          available here.
        </Text>
      </View>

      <View style={styles.option}>
        <Text style={styles.optionTitle}>Speech & Voice</Text>
        <Text style={styles.optionText}>
          Speech speed and volume controls will be available here.
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#08090B',
    padding: 24,
    paddingTop: 60,
  },
  back: {
    color: '#08C8F8',
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 30,
  },
  title: {
    color: '#FFFFFF',
    fontSize: 30,
    fontWeight: '800',
    marginBottom: 30,
  },
  option: {
    backgroundColor: '#17191E',
    borderRadius: 14,
    padding: 20,
    marginBottom: 15,
  },
  optionTitle: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '700',
  },
  optionText: {
    color: '#A7A9AE',
    fontSize: 14,
    lineHeight: 20,
    marginTop: 8,
  },
  input: {
    marginTop: 12,
    backgroundColor: '#08090B',
    borderWidth: 1,
    borderColor: '#2C2E35',
    borderRadius: 10,
    color: '#FFFFFF',
    fontSize: 15,
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  saveButton: {
    backgroundColor: '#08C8F8',
    borderRadius: 13,
    padding: 16,
    alignItems: 'center',
    marginBottom: 15,
  },
  saveText: {
    color: '#000000',
    fontSize: 13,
    fontWeight: '800',
  },
  status: {
    color: '#20D080',
    fontSize: 13,
    marginBottom: 15,
  },
});
