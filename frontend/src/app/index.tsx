import { useRouter } from 'expo-router';
import {
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

export default function HomeScreen() {
  const router = useRouter();

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>

        {/* HEADER */}
        <View style={styles.header}>
          <Text style={styles.logo}>BlindSight</Text>

          <Pressable
            style={styles.settingsButton}
            onPress={() => router.push('/settings')}
            accessibilityRole="button"
            accessibilityLabel="Settings"
          >
            <Text style={styles.settingsIcon}>⚙</Text>
          </Pressable>
        </View>

        {/* INTRODUCTION */}
        <View style={styles.intro}>
          <Text style={styles.title}>
            Your surroundings,{'\n'}
            without the noise.
          </Text>

          <Text style={styles.subtitle}>
            Select an option below. BlindSight describes what
            you looked at through a short camera capture.
          </Text>
        </View>

        {/* MODES */}
        <View style={styles.modes}>

          {/* ORIENT ME */}
          <Pressable
            style={styles.primaryButton}
            onPress={() => router.push('/orient')}
            accessibilityRole="button"
            accessibilityLabel="Orient me"
            accessibilityHint="Continuously monitors your surroundings"
          >
            <View style={styles.buttonIconCircle}>
              <Text style={styles.buttonIcon}>◉</Text>
            </View>

            <View style={styles.buttonContent}>
              <Text style={styles.buttonTitle}>
                ORIENT ME
              </Text>

              <Text style={styles.buttonSubtitle}>
                Continuously monitor updates when your
                environment changes
              </Text>
            </View>

            <Text style={styles.arrow}>›</Text>
          </Pressable>

          {/* DESCRIBE THIS PLACE */}
          <Pressable
            style={styles.secondaryButton}
            onPress={() => router.push('/describe')}
            accessibilityRole="button"
            accessibilityLabel="Describe this place"
            accessibilityHint="Get a detailed description of your surroundings"
          >
            <View style={styles.buttonIconCircleDark}>
              <Text style={styles.buttonIcon}>◉</Text>
            </View>

            <View style={styles.buttonContent}>
              <Text style={styles.buttonTitle}>
                DESCRIBE THIS PLACE
              </Text>

              <Text style={styles.buttonSubtitle}>
                Get a description of what the
                camera saw
              </Text>
            </View>

            <Text style={styles.arrow}>›</Text>
          </Pressable>

          {/* DEMO */}
          <Pressable
            style={styles.secondaryButton}
            onPress={() => router.push('/demo')}
            accessibilityRole="button"
            accessibilityLabel="Demo"
            accessibilityHint="Choose a demonstration scene"
          >
            <View style={styles.buttonIconCircleDark}>
              <Text style={styles.buttonIcon}>▶</Text>
            </View>

            <View style={styles.buttonContent}>
              <Text style={styles.buttonTitle}>
                DEMO
              </Text>

              <Text style={styles.buttonSubtitle}>
                Analyse preloaded demonstration scenes
              </Text>
            </View>

            <Text style={styles.arrow}>›</Text>
          </Pressable>

        </View>


      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#08090B',
  },

  content: {
    flex: 1,
    paddingHorizontal: 20,
    paddingTop: 12,
    paddingBottom: 20,
  },

  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },

  logo: {
    color: '#FFFFFF',
    fontSize: 20,
    fontWeight: '800',
    letterSpacing: 0.3,
  },

  settingsButton: {
    width: 42,
    height: 42,
    borderRadius: 12,
    backgroundColor: '#191A1F',
    alignItems: 'center',
    justifyContent: 'center',
  },

  settingsIcon: {
    color: '#FFFFFF',
    fontSize: 20,
  },

  intro: {
    marginTop: 55,
    marginBottom: 28,
  },

  title: {
    color: '#FFFFFF',
    fontSize: 30,
    lineHeight: 35,
    fontWeight: '800',
  },

  subtitle: {
    color: '#92959D',
    fontSize: 14,
    lineHeight: 21,
    marginTop: 12,
    maxWidth: 350,
  },

  modes: {
    gap: 12,
  },

  primaryButton: {
    minHeight: 92,
    borderRadius: 16,
    backgroundColor: '#08C8F8',
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
  },

  secondaryButton: {
    minHeight: 92,
    borderRadius: 16,
    backgroundColor: '#191A1F',
    borderWidth: 1,
    borderColor: '#2C2E35',
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
  },

  buttonIconCircle: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#FFFFFF',
    alignItems: 'center',
    justifyContent: 'center',
  },

  buttonIconCircleDark: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#292B31',
    alignItems: 'center',
    justifyContent: 'center',
  },

  buttonIcon: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '800',
  },

  buttonContent: {
    flex: 1,
    marginLeft: 14,
    marginRight: 8,
  },

  buttonTitle: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '800',
    letterSpacing: 0.3,
  },

  buttonSubtitle: {
    color: '#A7A9AE',
    fontSize: 12,
    lineHeight: 17,
    marginTop: 4,
  },

  arrow: {
    color: '#FFFFFF',
    fontSize: 28,
    fontWeight: '300',
  },

  demoShortcut: {
    alignSelf: 'center',
    marginTop: 'auto',
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 14,
    paddingVertical: 9,
    borderRadius: 20,
    backgroundColor: '#111216',
  },

  demoShortcutIcon: {
    color: '#08C8F8',
    fontSize: 11,
    marginRight: 6,
  },

  demoShortcutText: {
    color: '#08C8F8',
    fontSize: 12,
    fontWeight: '700',
  },
});