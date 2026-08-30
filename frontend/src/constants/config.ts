import { Platform } from 'react-native';

const DEFAULT_API_URL = process.env.EXPO_PUBLIC_BLINDSIGHT_API_URL ?? '';
const DEFAULT_API_KEY = process.env.EXPO_PUBLIC_BLINDSIGHT_API_KEY ?? '';

let runtimeUrl = DEFAULT_API_URL.replace(/\/$/, '');
let runtimeKey = DEFAULT_API_KEY;

export function getApiUrl(): string {
  if (runtimeUrl) {
    return runtimeUrl;
  }
  if (Platform.OS === 'web' && typeof window !== 'undefined') {
    return window.location.origin;
  }
  return '';
}

export function getApiKey(): string {
  return runtimeKey;
}

export function setApiUrl(url: string): void {
  runtimeUrl = url.trim().replace(/\/$/, '');
}

export function setApiKey(key: string): void {
  runtimeKey = key.trim();
}

export function resolveUrl(pathOrUrl: string): string {
  if (/^https?:\/\//i.test(pathOrUrl)) {
    return pathOrUrl;
  }
  const base = getApiUrl();
  if (!base) {
    return pathOrUrl;
  }
  return `${base}${pathOrUrl.startsWith('/') ? pathOrUrl : `/${pathOrUrl}`}`;
}
