import { getApiKey, resolveUrl } from '@/constants/config';
import {
  ApiError,
  completeCapture,
  createLiveCapture,
  pollCapture,
  type CaptureResource,
  type ErrorObject,
  type LiveMimeType,
} from '@/services/api';
import { File, Paths, UploadType } from 'expo-file-system';
import { Platform } from 'react-native';

// ~1 MiB chunks pipeline better over mobile uplink and retry cheaply, and a
// bounded pool hides per-request latency without saturating a weak uplink.
const CHUNK_BYTES = 1024 * 1024;
const UPLOAD_CONCURRENCY = 3;

export function captureMimeType(): LiveMimeType {
  return Platform.OS === 'ios' ? 'video/quicktime' : 'video/mp4';
}

// Native fetch and expo/fetch cannot send ArrayBufferView bodies on RN
// ("Creating blobs from ArrayBuffer and ArrayBufferView are not supported").
// File.upload with UploadType.BINARY_CONTENT sends the file bytes as the raw
// request body and passes our headers through untouched (verified against the
// Android and iOS native implementations). OpenAPI wants
// application/octet-stream on PUT /chunks/{index}.
async function putCaptureChunkFile(
  captureId: string,
  index: number,
  fileUri: string,
): Promise<void> {
  const result = await new File(fileUri).upload(
    resolveUrl(`/v1/captures/${captureId}/chunks/${index}`),
    {
      httpMethod: 'PUT',
      uploadType: UploadType.BINARY_CONTENT,
      headers: {
        'X-API-Key': getApiKey(),
        'Content-Type': 'application/octet-stream',
      },
    },
  );
  if (result.status >= 400) {
    let error: ErrorObject = {
      code: 'HTTP_ERROR',
      message: `Chunk upload failed (${result.status}).`,
      retryable: result.status >= 500,
      details: {},
    };
    try {
      const body = JSON.parse(result.body) as { error?: ErrorObject };
      if (body.error) {
        error = body.error;
      }
    } catch {
      // Non-JSON error body; keep the status fallback.
    }
    throw new ApiError(result.status, error);
  }
}

async function uploadRecordingInChunks(
  captureId: string,
  recordedUri: string,
  total: number,
): Promise<number> {
  if (total <= CHUNK_BYTES) {
    await putCaptureChunkFile(captureId, 0, recordedUri);
    return 1;
  }

  const source = new File(recordedUri);
  const handle = source.open();
  const chunkCount = Math.ceil(total / CHUNK_BYTES);
  const chunkFiles: File[] = [];
  try {
    for (let index = 0; index < chunkCount; index++) {
      const remaining = total - index * CHUNK_BYTES;
      const length = Math.min(CHUNK_BYTES, remaining);
      const slice = handle.readBytes(length);
      const chunkFile = new File(
        Paths.cache,
        `blindsight-chunk-${captureId}-${index}.bin`,
      );
      chunkFile.create({ overwrite: true });
      chunkFile.write(slice);
      chunkFiles.push(chunkFile);
    }
  } finally {
    handle.close();
  }

  try {
    // Arrival order is irrelevant and repeating identical bytes at an index
    // is idempotent (OpenAPI PUT /chunks/{index}), so a bounded parallel
    // pool is safe. All uploads settle before temp files are cleaned up.
    let nextIndex = 0;
    const uploadNext = (): Promise<void> => {
      if (nextIndex >= chunkCount) {
        return Promise.resolve();
      }
      const index = nextIndex++;
      return putCaptureChunkFile(captureId, index, chunkFiles[index].uri).then(
        uploadNext,
      );
    };
    const workers = Array.from(
      { length: Math.min(UPLOAD_CONCURRENCY, chunkCount) },
      uploadNext,
    );
    const results = await Promise.allSettled(workers);
    const failure = results.find(
      (result): result is PromiseRejectedResult => result.status === 'rejected',
    );
    if (failure) {
      throw failure.reason;
    }
  } finally {
    for (const chunkFile of chunkFiles) {
      try {
        if (chunkFile.exists) {
          chunkFile.delete();
        }
      } catch {
        // A leftover temp chunk in cache is harmless.
      }
    }
  }
  return chunkCount;
}

// Starts the POST /v1/captures round trip in the background so it overlaps
// the recording interval instead of standing between recording and upload.
export function prepareLiveCapture(): Promise<CaptureResource> {
  return createLiveCapture(captureMimeType());
}

// Closes the scene session of a prepared capture that was never used.
export async function discardPreparedCapture(
  preparedCapture: Promise<CaptureResource>,
): Promise<void> {
  try {
    const created = await preparedCapture;
    await deleteSceneSessionQuietly(created.scene_session_id);
  } catch {
    // Creation failed or cleanup did; the session is over for this client.
  }
}

export async function runLiveCapture(
  recordedUri: string,
  preparedCapture?: Promise<CaptureResource>,
): Promise<CaptureResource> {
  const mime = captureMimeType();
  const created = await (preparedCapture ?? createLiveCapture(mime));
  try {
    const file = new File(recordedUri);
    const total = file.size;
    if (total <= 0) {
      throw new ApiError(0, {
        code: 'CLIENT_EMPTY_RECORDING',
        message: 'The recording was empty. Try again.',
        retryable: true,
        details: {},
      });
    }
    const chunkCount = await uploadRecordingInChunks(
      created.capture_id,
      recordedUri,
      total,
    );
    await completeCapture(created.capture_id, chunkCount, mime);
  } catch (err) {
    deleteSceneSessionQuietly(created.scene_session_id);
    throw err;
  }
  return await pollCapture(created.capture_id);
}

async function deleteSceneSessionQuietly(sceneSessionId: string): Promise<void> {
  try {
    const { deleteSceneSession } = await import('@/services/api');
    await deleteSceneSession(sceneSessionId);
  } catch {
    // The session is over for this client either way.
  }
}
