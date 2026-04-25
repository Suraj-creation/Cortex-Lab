import React, { createContext, useContext, useMemo } from 'react';
import { useNetInfo } from '@react-native-community/netinfo';

interface NetworkStatus {
  isConnected: boolean | null;
  isInternetReachable: boolean | null;
  isOffline: boolean;
  connectionType: string;
}

const NetworkContext = createContext<NetworkStatus>({
  isConnected: null,
  isInternetReachable: null,
  isOffline: false,
  connectionType: 'unknown',
});

export function NetworkProvider({ children }: { children: React.ReactNode }) {
  const netInfo = useNetInfo();

  const value = useMemo<NetworkStatus>(() => {
    const isConnected = typeof netInfo.isConnected === 'boolean' ? netInfo.isConnected : null;
    const isInternetReachable =
      typeof netInfo.isInternetReachable === 'boolean' ? netInfo.isInternetReachable : null;

    const isOffline = isConnected === false || isInternetReachable === false;

    return {
      isConnected,
      isInternetReachable,
      isOffline,
      connectionType: netInfo.type || 'unknown',
    };
  }, [netInfo.isConnected, netInfo.isInternetReachable, netInfo.type]);

  return <NetworkContext.Provider value={value}>{children}</NetworkContext.Provider>;
}

export function useNetworkStatus() {
  return useContext(NetworkContext);
}
