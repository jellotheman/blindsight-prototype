import {
  ApiError,
  deleteSceneSession,
  type CaptureResource,
} from '@/services/api';
import {
  discardPreparedCapture,
  prepareLiveCapture,
  runLiveCapture,
} from '@/services/capture';
import { CameraView, useCameraPermissions, useMicrophonePermissions } from 'expo-camera';
import { useRouter } from 'expo-router';
import * as Speech from 'expo-speech';
import { useRef, useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

type DescribeState = 'idle' | 'recording' | 'processing' | 'result';

export default function DescribeScreen() {
  const router = useRouter();
  const insets = useSafeAreaInsets();
  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  const [micPermission, requestMicPermission] = useMicrophonePermissions();

  const cameraRef = useRef<CameraView>(null);
  const sceneSessionId = useRef<string | null>(null);

  const [state, setState] = useState<DescribeState>('idle');
  const [overview, setOverview] = useState('');
  const [placeType, setPlaceType] = useState('');
  const [errorMessage, setErrorMessage] = useState('');

  const requestPermissions = () => {
    requestCameraPermission();
    requestMicPermission();
  };

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

  const startCapture = async () => {
    if (state === 'recording' || state === 'processing') {
      return;
    }
    Speech.stop();
    setErrorMessage('');
    setState('recording');
    // Overlap the capture-session round trip with the recording interval; the
    // prepared capture is handed to runLiveCapture once recording finishes.
    const preparedCapture = prepareLiveCapture();
    try {
      const recording = await cameraRef.current?.recordAsync({
        maxDuration: 8,
      });
      if (!recording?.uri) {
        throw new ApiError(0, {
          code: 'CLIENT_NO_RECORDING',
          message: 'Could not record video on this device.',
          retryable: true,
          details: {},
        });
      }
      setState('processing');
      const settled: CaptureResource = await runLiveCapture(
        recording.uri,
        preparedCapture,
      );
      if (settled.status === 'succeeded' && settled.card) {
        const card = settled.card.card;
        sceneSessionId.current = settled.scene_session_id;
        setOverview(card.overview);
        setPlaceType(card.place_type ?? '');
        setState('result');
        Speech.speak(card.overview, { rate: 0.9 });
      } else {
        setErrorMessage(settled.failure?.message ?? 'The capture failed.');
        setState('result');
      }
    } catch (err: unknown) {
      // If recording failed, the prepared capture was never used; if upload
      // failed, runLiveCapture already closed it (idempotent either way).
      void discardPreparedCapture(preparedCapture);
      setErrorMessage(err instanceof Error ? err.message : 'The capture failed.');
      setState('result');
    }
  };

  const replay = () => {
    Speech.stop();
    if (overview) {
      Speech.speak(overview, { rate: 0.9 });
    }
  };

  const newCapture = () => {
    Speech.stop();
    void closeSession();
    setOverview('');
    setPlaceType('');
    setErrorMessage('');
    setState('idle');
  };

  const goBack = () => {
    Speech.stop();
    void closeSession();
    router.back();
  };

  const askQuestion = () => {
    Speech.stop();
    router.push({
      pathname: '/chat',
      params: { sceneSessionId: sceneSessionId.current ?? '' },
    });
  };

  if (!cameraPermission || !micPermission) {
    return <View style={styles.container} />;
  }

  if (!cameraPermission.granted || !micPermission.granted) {
    return (
      <View style={styles.container}>
        <View style={styles.permissionContainer}>
          <Text style={styles.title}>Camera & Microphone</Text>

            <Text style={styles.description}>
              BlindSight needs your camera and microphone
              to capture and describe a view.
            </Text>

          <Pressable
            style={styles.primaryButton}
            onPress={requestPermissions}
          >
            <Text style={styles.primaryText}>ALLOW ACCESS</Text>
          </Pressable>

          <Pressable onPress={() => router.back()}>
            <Text style={styles.backText}>GO BACK</Text>
          </Pressable>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.cameraContainer}>
        <CameraView
          ref={cameraRef}
          style={styles.camera}
          facing="back"
          mode="video"
          videoQuality="480p"
          videoBitrate={1_000_000}
          mute={false}
        />

        <View style={[styles.topBar, { top: insets.top + 10 }]}>
          <Pressable onPress={goBack}>
            <Text style={styles.back}>‹ Back</Text>
          </Pressable>

          <Text style={styles.mode}>DESCRIBE THIS PLACE</Text>

          <View style={{ width: 45 }} />
        </View>
      </View>

      <View style={styles.bottomPanel}>
        {state === 'idle' && (
          <>
            <Text style={styles.heading}>
              Ready when{'\n'}you are
            </Text>

            <Text style={styles.description}>
              One capture lasts about eight seconds. Point
              the camera and start when ready.
            </Text>

            <Pressable
              style={styles.primaryButton}
              onPress={startCapture}
              accessibilityRole="button"
              accessibilityLabel="Start capture"
              accessibilityHint="Records about eight seconds and builds a description"
            >
              <Text style={styles.primaryText}>START CAPTURE</Text>
            </Pressable>
          </>
        )}

        {state === 'recording' && (
          <>
            <View style={styles.redBadge}>
              <Text style={styles.redText}>● CAPTURING</Text>
            </View>

            <Text style={styles.heading}>
              Capturing your{'\n'}surroundings...
            </Text>

            <Text style={styles.description}>
              Look or point the camera around. The capture ends on its own.
            </Text>
          </>
        )}

        {state === 'processing' && (
          <>
            <View style={styles.blueBadge}>
              <Text style={styles.blueText}>AI PROCESSING</Text>
            </View>

            <Text style={styles.heading}>
              Understanding what{'\n'}you looked at...
            </Text>

            <Text style={styles.description}>
              This may take several seconds. Stay put if you can.
            </Text>
          </>
        )}

        {state === 'result' && (
          <>
            {errorMessage ? (
              <>
                <View style={styles.redBadge}>
                  <Text style={styles.redText}>✕ FAILED</Text>
                </View>

                <Text style={styles.description}>{errorMessage}</Text>
              </>
            ) : (
              <>
                <View style={styles.greenBadge}>
                  <Text style={styles.greenText}>✓ READY</Text>
                </View>

                {placeType ? (
                  <Text style={styles.placeType}>{placeType.toUpperCase()}</Text>
                ) : null}

                <Text style={styles.description}>{overview}</Text>
              </>
            )}

            {!errorMessage ? (
              <View style={styles.actionRow}>
                <Pressable style={styles.smallButton} onPress={replay}>
                  <Text style={styles.smallButtonText}>REPEAT</Text>
                </Pressable>

                <Pressable
                  style={styles.smallButton}
                  onPress={() => {
                    Speech.speak('What would you like to know?', { rate: 0.9 });
                    askQuestion();
                  }}
                >
                  <Text style={styles.smallButtonText}>ASK A QUESTION</Text>
                </Pressable>
              </View>
            ) : null}

            <Pressable style={styles.cancelButton} onPress={newCapture}>
              <Text style={styles.cancelText}>NEW CAPTURE</Text>
            </Pressable>
          </>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#08090B',
  },

  permissionContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 28,
  },

  cameraContainer: {
    flex: 1,
  },

  camera: {
    flex: 1,
  },

  topBar: {
    position: 'absolute',
    top: 45,
    left: 20,
    right: 20,
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

  bottomPanel: {
    backgroundColor: '#111216',
    padding: 22,
    paddingBottom: 28,
  },

  title: {
    color: '#FFFFFF',
    fontSize: 28,
    fontWeight: '800',
    textAlign: 'center',
  },

  description: {
    color: '#A7A9AE',
    fontSize: 15,
    lineHeight: 22,
    marginTop: 12,
    textAlign: 'center',
  },

  heading: {
    color: '#FFFFFF',
    fontSize: 25,
    lineHeight: 30,
    fontWeight: '800',
    marginTop: 14,
  },

  placeType: {
    color: '#08C8F8',
    fontSize: 13,
    fontWeight: '800',
    letterSpacing: 1,
    marginTop: 14,
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

  greenBadge: {
    alignSelf: 'flex-start',
    borderWidth: 1,
    borderColor: '#20D080',
    borderRadius: 20,
    paddingHorizontal: 12,
    paddingVertical: 6,
  },

  greenText: {
    color: '#20D080',
    fontSize: 11,
    fontWeight: '800',
  },

  redBadge: {
    alignSelf: 'flex-start',
    borderWidth: 1,
    borderColor: '#FF6B6B',
    borderRadius: 20,
    paddingHorizontal: 12,
    paddingVertical: 6,
  },

  redText: {
    color: '#FF6B6B',
    fontSize: 11,
    fontWeight: '800',
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

  backText: {
    color: '#08C8F8',
    fontSize: 14,
    fontWeight: '700',
    marginTop: 22,
  },

  actionRow: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 20,
  },

  smallButton: {
    flex: 1,
    backgroundColor: '#24262D',
    padding: 14,
    borderRadius: 13,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#3A3D45',
  },

  smallButtonText: {
    color: '#FFFFFF',
    fontSize: 13,
    fontWeight: '800',
  },

  cancelButton: {
    backgroundColor: '#24262D',
    padding: 16,
    borderRadius: 13,
    marginTop: 14,
    alignItems: 'center',
  },

  cancelText: {
    color: '#FFFFFF',
    fontSize: 13,
    fontWeight: '800',
  },
});
