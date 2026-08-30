import { useRouter } from 'expo-router';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

// Orient Me is continuous monitoring of your surroundings. The transition model it
// depends on does not exist yet, so this mode is a stub: it makes no capture and no claim.
// It will light up once the transition model is ready.
export default function OrientScreen() {
  const router = useRouter();
  const insets = useSafeAreaInsets();

  return (
    <View style={styles.container}>
      <View style={[styles.topBar, { paddingTop: insets.top + 10 }]}>
        <Pressable onPress={() => router.back()}>
          <Text style={styles.back}>‹ Back</Text>
        </Pressable>

        <Text style={styles.mode}>ORIENT ME</Text>

        <View style={{ width: 45 }} />
      </View>

      <View style={styles.content}>
        <View style={styles.badge}>
          <Text style={styles.badgeText}>COMING SOON</Text>
        </View>

        <Text style={styles.heading}>
          Not available{'\n'}yet
        </Text>

        <Text style={styles.description}>
          Continuous orientation is not built yet. To understand a place, use
          Describe This Place instead — it captures for a few seconds and
          describes what it saw.
        </Text>

        <Pressable
          style={styles.primaryButton}
          onPress={() => router.replace('/describe')}
          accessibilityRole="button"
          accessibilityLabel="Use describe this place instead"
          accessibilityHint="Opens the describe mode"
        >
          <Text style={styles.primaryText}>USE DESCRIBE INSTEAD</Text>
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#08090B',
  },

  topBar: {
    paddingHorizontal: 20,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },

  back: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '700',
  },

  mode: {
    color: '#FFFFFF',
    fontSize: 13,
    fontWeight: '800',
    letterSpacing: 1,
  },

  content: {
    flex: 1,
    justifyContent: 'center',
    padding: 28,
  },

  badge: {
    alignSelf: 'flex-start',
    borderWidth: 1,
    borderColor: '#FFC400',
    borderRadius: 20,
    paddingHorizontal: 12,
    paddingVertical: 6,
  },

  badgeText: {
    color: '#FFC400',
    fontSize: 11,
    fontWeight: '800',
  },

  heading: {
    color: '#FFFFFF',
    fontSize: 28,
    fontWeight: '800',
    marginTop: 18,
  },

  description: {
    color: '#A7A9AE',
    fontSize: 15,
    lineHeight: 22,
    marginTop: 12,
  },

  primaryButton: {
    backgroundColor: '#08C8F8',
    paddingHorizontal: 28,
    paddingVertical: 17,
    borderRadius: 14,
    marginTop: 25,
    alignItems: 'center',
  },

  primaryText: {
    color: '#000000',
    fontSize: 15,
    fontWeight: '800',
  },
});
