import { loadCredentials } from '@/services/credentials';
import { DarkTheme, Stack, ThemeProvider } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { useEffect, useState } from 'react';
import { View } from 'react-native';

export default function RootLayout() {
  const [ready, setReady] = useState(false);

  useEffect(() => {
    loadCredentials().finally(() => setReady(true));
  }, []);

  if (!ready) {
    return <View style={{ flex: 1, backgroundColor: '#08090B' }} />;
  }

  return (
    <ThemeProvider value={DarkTheme}>
      <Stack
        screenOptions={{
          headerShown: false,
          contentStyle: {
            backgroundColor: '#08090B',
          },
          animation: 'none',
        }}
      />
      <StatusBar style="light" />
    </ThemeProvider>
  );
}
