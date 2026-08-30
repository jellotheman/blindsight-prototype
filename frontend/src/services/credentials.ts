import * as SecureStore from 'expo-secure-store';

import { getApiKey, getApiUrl, setApiKey, setApiUrl } from '@/constants/config';

const URL_KEY = 'blindsight_api_url';
const API_KEY = 'blindsight_api_key';

let loaded = false;

async function read(key: string): Promise<string | null> {
  try {
    if (!(await SecureStore.isAvailableAsync())) {
      return null;
    }
    return await SecureStore.getItemAsync(key);
  } catch {
    return null;
  }
}

async function write(key: string, value: string): Promise<void> {
  try {
    if (!(await SecureStore.isAvailableAsync())) {
      return;
    }
    await SecureStore.setItemAsync(key, value);
  } catch {
    // In-memory config still applies for this session.
  }
}

export async function loadCredentials(): Promise<void> {
  if (loaded) {
    return;
  }
  const [storedUrl, storedKey] = await Promise.all([read(URL_KEY), read(API_KEY)]);
  if (storedUrl) {
    setApiUrl(storedUrl);
  }
  if (storedKey) {
    setApiKey(storedKey);
  }
  loaded = true;
}

export async function saveCredentials(url: string, key: string): Promise<void> {
  setApiUrl(url);
  setApiKey(key);
  await Promise.all([write(URL_KEY, getApiUrl()), write(API_KEY, getApiKey())]);
  loaded = true;
}

export function currentCredentials(): { url: string; key: string } {
  return { url: getApiUrl(), key: getApiKey() };
}
