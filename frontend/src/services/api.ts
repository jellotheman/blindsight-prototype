import { getApiKey, getApiUrl, resolveUrl } from '@/constants/config';

export type CaptureStatus = 'recording' | 'processing' | 'succeeded' | 'failed';
export type QuestionStatus =
  | 'processing'
  | 'answered'
  | 'needs_clip_consent'
  | 'unanswerable'
  | 'failed';
export type AnswerSource = 'scene_card' | 'captured_view';
export type Confidence = 'high' | 'medium' | 'low';
export type Distance = 'close' | 'middle' | 'far';
export type LiveMimeType = 'video/webm' | 'video/mp4' | 'video/quicktime';

export type ErrorObject = {
  code: string;
  message: string;
  retryable: boolean;
  details: Record<string, unknown>;
};

export class ApiError extends Error {
  code: string;
  retryable: boolean;
  details: Record<string, unknown>;
  status: number;

  constructor(status: number, error: ErrorObject) {
    super(error.message);
    this.name = 'ApiError';
    this.status = status;
    this.code = error.code;
    this.retryable = error.retryable;
    this.details = error.details;
  }
}

export type Excerpt = {
  excerpt_id: string;
  label: string;
  duration_seconds: number;
  poster_url: string;
};

export type LiveSource = { type: 'live'; mime_type: LiveMimeType };
export type ExcerptSource = { type: 'excerpt'; excerpt_id: string };
export type CaptureSource = LiveSource | ExcerptSource;

export type AsyncFailure = {
  code: string;
  message: string;
  retryable: boolean;
};

export type LayoutItem = {
  thing: string;
  relationship: string | null;
  distance: Distance | null;
  confidence: Confidence;
};

export type PersonObservation = {
  count_description: string;
  relationship: string | null;
  activity: string | null;
  confidence: Confidence;
};

export type ClaimUncertainty = {
  claim: string;
  detail: string;
};

export type SceneCardBody = {
  place_type: string | null;
  place_type_confidence: Confidence | null;
  overview: string;
  layout: LayoutItem[] | null;
  open_space: string | null;
  people: PersonObservation[] | null;
  visual_character: string | null;
  uncertainties: ClaimUncertainty[] | null;
};

export type SceneCard = {
  capture_id: string;
  scene_session_id: string;
  revision: number;
  evidence: string[];
  card: SceneCardBody;
};

export type CaptureResource = {
  capture_id: string;
  scene_session_id: string;
  source: CaptureSource;
  status: CaptureStatus;
  card: SceneCard | null;
  failure: AsyncFailure | null;
  created_at: string;
  updated_at: string;
};

export type QuestionResource = {
  question_id: string;
  scene_session_id: string;
  question: string;
  status: QuestionStatus;
  answer: string | null;
  source: AnswerSource | null;
  failure: AsyncFailure | null;
  created_at: string;
  updated_at: string;
};

const DEFAULT_POLL_MS = 1000;
const CAPTURE_POLL_DEADLINE_MS = 105_000;
const QUESTION_POLL_DEADLINE_MS = 75_000;
const CAPTURE_SETTLED: CaptureStatus[] = ['succeeded', 'failed'];
const QUESTION_SETTLED: QuestionStatus[] = [
  'answered',
  'needs_clip_consent',
  'unanswerable',
  'failed',
];

function requireConfig(): { key: string } {
  const base = getApiUrl();
  const key = getApiKey();
  if (!base || !key) {
    throw new ApiError(0, {
      code: 'CLIENT_NOT_CONFIGURED',
      message: 'Set the API URL and key in Settings first.',
      retryable: false,
      details: {},
    });
  }
  return { key };
}

async function request(path: string, options: RequestInit = {}): Promise<Response> {
  const { key } = requireConfig();
  const headers = new Headers(options.headers);
  headers.set('X-API-Key', key);
  const response = await fetch(resolveUrl(path), { ...options, headers });
  if (!response.ok) {
    let error: ErrorObject = {
      code: 'HTTP_ERROR',
      message: `Request failed (${response.status}).`,
      retryable: response.status >= 500,
      details: {},
    };
    try {
      const body = (await response.json()) as { error?: ErrorObject };
      if (body.error) {
        error = body.error;
      }
    } catch {
      // Non-JSON error body; keep the status fallback.
    }
    throw new ApiError(response.status, error);
  }
  return response;
}

async function requestJson<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await request(path, options);
  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

function retryAfterMs(response: Response): number {
  const header = response.headers.get('Retry-After');
  const seconds = header ? Number(header) : NaN;
  return Number.isFinite(seconds) && seconds > 0 ? seconds * 1000 : DEFAULT_POLL_MS;
}

async function pollUntilSettled<T extends { status: string }>(
  path: string,
  settled: readonly string[],
  deadlineMs: number,
): Promise<T> {
  const deadline = Date.now() + deadlineMs;
  for (;;) {
    const response = await request(path);
    const resource = (await response.json()) as T;
    if (settled.includes(resource.status)) {
      return resource;
    }
    if (Date.now() > deadline) {
      throw new ApiError(0, {
        code: 'CLIENT_POLL_TIMEOUT',
        message: 'The request did not settle in time.',
        retryable: true,
        details: {},
      });
    }
    await new Promise((resolve) => setTimeout(resolve, retryAfterMs(response)));
  }
}

export async function listExcerpts(): Promise<Excerpt[]> {
  const body = await requestJson<{ items: Excerpt[] }>('/v1/excerpts');
  return body.items;
}

export async function createExcerptCapture(excerptId: string): Promise<CaptureResource> {
  return requestJson<CaptureResource>('/v1/captures', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source: { type: 'excerpt', excerpt_id: excerptId } }),
  });
}

export async function createLiveCapture(mimeType: LiveMimeType): Promise<CaptureResource> {
  return requestJson<CaptureResource>('/v1/captures', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source: { type: 'live', mime_type: mimeType } }),
  });
}

export async function completeCapture(
  captureId: string,
  chunkCount: number,
  mimeType: LiveMimeType,
): Promise<CaptureResource> {
  return requestJson<CaptureResource>(`/v1/captures/${captureId}/complete`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ chunk_count: chunkCount, mime_type: mimeType }),
  });
}

export async function getCapture(captureId: string): Promise<CaptureResource> {
  return requestJson<CaptureResource>(`/v1/captures/${captureId}`);
}

export async function pollCapture(captureId: string): Promise<CaptureResource> {
  return pollUntilSettled<CaptureResource>(
    `/v1/captures/${captureId}`,
    CAPTURE_SETTLED,
    CAPTURE_POLL_DEADLINE_MS,
  );
}

export async function createQuestion(
  sceneSessionId: string,
  question: string,
): Promise<QuestionResource> {
  return requestJson<QuestionResource>(`/v1/scene-sessions/${sceneSessionId}/questions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  });
}

export async function pollQuestion(
  sceneSessionId: string,
  questionId: string,
): Promise<QuestionResource> {
  return pollUntilSettled<QuestionResource>(
    `/v1/scene-sessions/${sceneSessionId}/questions/${questionId}`,
    QUESTION_SETTLED,
    QUESTION_POLL_DEADLINE_MS,
  );
}

export async function checkCapturedView(
  sceneSessionId: string,
  questionId: string,
): Promise<QuestionResource> {
  return requestJson<QuestionResource>(
    `/v1/scene-sessions/${sceneSessionId}/questions/${questionId}/clip-check`,
    { method: 'POST' },
  );
}

export async function deleteSceneSession(sceneSessionId: string): Promise<void> {
  await request(`/v1/scene-sessions/${sceneSessionId}`, { method: 'DELETE' });
}

export function posterSource(posterUrl: string): { uri: string; headers: { 'X-API-Key': string } } {
  return {
    uri: resolveUrl(posterUrl),
    headers: { 'X-API-Key': getApiKey() },
  };
}
