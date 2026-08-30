import {
  ApiError,
  createExcerptCapture,
  deleteSceneSession,
  listExcerpts,
  pollCapture,
  posterSource,
  type Excerpt,
} from '@/services/api';
import { Image } from 'expo-image';
import { useRouter } from 'expo-router';
import * as Speech from 'expo-speech';
import { useEffect, useRef, useState } from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

type DemoState = 'select' | 'processing' | 'result';

export default function DemoScreen() {
  const router = useRouter();
  const insets = useSafeAreaInsets();
  const [state, setState] = useState<DemoState>('select');
  const [excerpts, setExcerpts] = useState<Excerpt[]>([]);
  const [loadError, setLoadError] = useState('');
  const [selectedScene, setSelectedScene] = useState<Excerpt | null>(null);
  const [description, setDescription] = useState('');
  const sceneSessionId = useRef<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    listExcerpts()
      .then((items) => {
        if (!cancelled) {
          setExcerpts(items);
          setLoadError('');
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setLoadError(err instanceof Error ? err.message : 'Could not load excerpts.');
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const closeSession = async () => {
    const id = sceneSessionId.current;
    sceneSessionId.current = null;
    if (!id) {
      return;
    }
    try {
      await deleteSceneSession(id);
    } catch {
      // Session is over for this client either way.
    }
  };

  const startDemo = async (scene: Excerpt) => {
    Speech.stop();
    setSelectedScene(scene);
    setDescription('');
    setState('processing');
    await closeSession();
    try {
      const created = await createExcerptCapture(scene.excerpt_id);
      sceneSessionId.current = created.scene_session_id;
      const settled = await pollCapture(created.capture_id);
      if (settled.status === 'succeeded' && settled.card) {
        const overview = settled.card.card.overview;
        setDescription(overview);
        setState('result');
        Speech.speak(overview, { rate: 0.9 });
      } else {
        const message = settled.failure?.message ?? 'The capture failed.';
        setDescription(message);
        setState('result');
      }
    } catch (err: unknown) {
      const message =
        err instanceof ApiError || err instanceof Error
          ? err.message
          : 'Could not run this demonstration.';
      setDescription(message);
      setState('result');
    }
  };

  const newScene = () => {
    Speech.stop();
    void closeSession();
    setDescription('');
    setSelectedScene(null);
    setState('select');
  };

  const replay = () => {
    Speech.stop();
    if (description) {
      Speech.speak(description, { rate: 0.9 });
    }
  };

  const goBack = () => {
    Speech.stop();
    void closeSession();
    router.back();
  };

  return (
    <View style={styles.container}>
      <View style={[styles.header, { height: insets.top + 48, paddingTop: insets.top + 6 }]}>
        <Pressable
          onPress={goBack}
          accessibilityRole="button"
          accessibilityLabel="Go back"
        >
          <Text style={styles.back}>‹ Back</Text>
        </Pressable>

        <Text style={styles.headerTitle}>DEMO</Text>

        <View style={{ width: 50 }} />
      </View>

      {state === 'select' && (
        <ScrollView contentContainerStyle={styles.selectionContainer}>
          <Text style={styles.title}>Choose a scene</Text>

          <Text style={styles.subtitle}>
            Select a demonstration video for BlindSight to analyse.
          </Text>

          {loadError ? <Text style={styles.error}>{loadError}</Text> : null}

          <View style={styles.grid}>
            {excerpts.map((scene) => (
              <Pressable
                key={scene.excerpt_id}
                style={styles.sceneCard}
                onPress={() => startDemo(scene)}
                accessibilityRole="button"
                accessibilityLabel={`Demo scene: ${scene.label}`}
                accessibilityHint="Starts analysing this demonstration scene"
              >
                <View style={styles.videoPlaceholder}>
                  <Image
                    source={posterSource(scene.poster_url)}
                    style={styles.poster}
                    contentFit="cover"
                  />
                  <View style={styles.playCircle}>
                    <Text style={styles.playIcon}>▶</Text>
                  </View>
                </View>

                <Text style={styles.sceneName}>{scene.label}</Text>

                <Text style={styles.duration}>{scene.duration_seconds}s</Text>
              </Pressable>
            ))}
          </View>
        </ScrollView>
      )}

      {state === 'processing' && (
        <View style={styles.processingContainer}>
          <View style={styles.videoLarge}>
            {selectedScene ? (
              <Image
                source={posterSource(selectedScene.poster_url)}
                style={styles.posterFill}
                contentFit="cover"
              />
            ) : null}
            <View style={styles.largePlayCircle}>
              <Text style={styles.largePlayIcon}>▶</Text>
            </View>
            <Text style={styles.videoLabel}>{selectedScene?.label.toUpperCase()}</Text>
          </View>

          <View style={styles.processingPanel}>
            <View style={styles.blueBadge}>
              <Text style={styles.blueText}>AI PROCESSING</Text>
            </View>

            <Text style={styles.heading}>
              Analysing this{'\n'}scene...
            </Text>

            <Text style={styles.descriptionText}>
              BlindSight is building a description of
              what it saw.
            </Text>

            <View style={styles.progressTrack}>
              <View style={styles.progressBar} />
            </View>
          </View>
        </View>
      )}

      {state === 'result' && (
        <ScrollView contentContainerStyle={styles.resultContainer}>
          <View style={styles.videoLarge}>
            {selectedScene ? (
              <Image
                source={posterSource(selectedScene.poster_url)}
                style={styles.posterFill}
                contentFit="cover"
              />
            ) : null}
            <View style={styles.largePlayCircle}>
              <Text style={styles.largePlayIcon}>▶</Text>
            </View>
            <Text style={styles.videoLabel}>{selectedScene?.label.toUpperCase()}</Text>
          </View>

          <View style={styles.resultPanel}>
            <View style={styles.blueBadge}>
              <Text style={styles.blueText}>● OVERVIEW</Text>
            </View>

            <Text style={styles.resultText}>{description}</Text>

            <View style={styles.actionRow}>
              <Pressable
                style={styles.largeActionButton}
                onPress={replay}
                accessibilityRole="button"
                accessibilityLabel="Repeat description"
                accessibilityHint="Plays the scene description again"
              >
                <Text style={styles.actionIcon}>▶</Text>
                <Text style={styles.actionLabel}>REPEAT</Text>
              </Pressable>

              <Pressable
                style={styles.largeActionButton}
                onPress={() => {
                  Speech.speak('What would you like to know?', { rate: 0.9 });
                  router.push({
                    pathname: '/chat',
                    params: { sceneSessionId: sceneSessionId.current ?? '' },
                  });
                }}
                accessibilityRole="button"
                accessibilityLabel="Ask a question about this scene"
                accessibilityHint="Opens the follow-up conversation"
              >
                <Text style={styles.actionIcon}>?</Text>
                <Text style={styles.actionLabel}>ASK</Text>
              </Pressable>

              <Pressable
                style={styles.largeActionButton}
                onPress={newScene}
                accessibilityRole="button"
                accessibilityLabel="Choose a new scene"
                accessibilityHint="Returns to the demonstration scene selection"
              >
                <Text style={styles.actionIcon}>↻</Text>
                <Text style={styles.actionLabel}>NEW SCENE</Text>
              </Pressable>
            </View>
          </View>
        </ScrollView>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#08090B',
  },
  header: {
    height: 90,
    paddingTop: 42,
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
  selectionContainer: {
    padding: 20,
    paddingBottom: 40,
  },
  title: {
    color: '#FFFFFF',
    fontSize: 28,
    fontWeight: '800',
    marginTop: 10,
  },
  subtitle: {
    color: '#A7A9AE',
    fontSize: 15,
    lineHeight: 22,
    marginTop: 8,
    marginBottom: 25,
  },
  error: {
    color: '#FF6B6B',
    fontSize: 14,
    lineHeight: 20,
    marginBottom: 16,
  },
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  sceneCard: {
    width: '31%',
    flexGrow: 1,
    backgroundColor: '#15161B',
    borderRadius: 14,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: '#2C2E35',
  },
  videoPlaceholder: {
    height: 105,
    backgroundColor: '#22242A',
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
  },
  poster: {
    width: '100%',
    height: '100%',
  },
  posterFill: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
  },
  playCircle: {
    position: 'absolute',
    width: 38,
    height: 38,
    borderRadius: 19,
    backgroundColor: '#08C8F8',
    alignItems: 'center',
    justifyContent: 'center',
  },
  playIcon: {
    color: '#000000',
    fontSize: 14,
    marginLeft: 2,
  },
  sceneName: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '800',
    paddingHorizontal: 9,
    paddingTop: 10,
  },
  duration: {
    color: '#8D9098',
    fontSize: 11,
    paddingHorizontal: 9,
    paddingTop: 3,
    paddingBottom: 12,
  },
  processingContainer: {
    flex: 1,
  },
  videoLarge: {
    flex: 1,
    backgroundColor: '#16171C',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 300,
  },
  largePlayCircle: {
    width: 75,
    height: 75,
    borderRadius: 38,
    borderWidth: 2,
    borderColor: '#08C8F8',
    alignItems: 'center',
    justifyContent: 'center',
  },
  largePlayIcon: {
    color: '#08C8F8',
    fontSize: 25,
    marginLeft: 3,
  },
  videoLabel: {
    color: '#666970',
    fontSize: 12,
    fontWeight: '800',
    letterSpacing: 1,
    marginTop: 15,
  },
  processingPanel: {
    backgroundColor: '#111216',
    padding: 22,
    paddingBottom: 30,
  },
  resultContainer: {
    paddingBottom: 40,
  },
  resultPanel: {
    backgroundColor: '#111216',
    padding: 22,
  },
  blueBadge: {
    alignSelf: 'flex-start',
    borderWidth: 1,
    borderColor: '#08C8F8',
    borderRadius: 20,
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  blueText: {
    color: '#08C8F8',
    fontSize: 11,
    fontWeight: '800',
  },
  heading: {
    color: '#FFFFFF',
    fontSize: 25,
    lineHeight: 30,
    fontWeight: '800',
    marginTop: 14,
  },
  descriptionText: {
    color: '#A7A9AE',
    fontSize: 15,
    lineHeight: 22,
    marginTop: 12,
  },
  progressTrack: {
    height: 4,
    backgroundColor: '#292B31',
    borderRadius: 5,
    marginTop: 20,
    overflow: 'hidden',
  },
  progressBar: {
    width: '65%',
    height: 4,
    backgroundColor: '#08C8F8',
  },
  resultText: {
    color: '#FFFFFF',
    fontSize: 17,
    lineHeight: 25,
    fontWeight: '600',
    marginTop: 16,
  },
  actionRow: {
    flexDirection: 'row',
    gap: 14,
    marginTop: 22,
  },
  largeActionButton: {
    flex: 1,
    minHeight: 105,
    backgroundColor: '#24262D',
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: '#3A3D45',
  },
  actionIcon: {
    color: '#FFFFFF',
    fontSize: 38,
    fontWeight: '700',
  },
  actionLabel: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '800',
    marginTop: 6,
    letterSpacing: 0.5,
  },
});
