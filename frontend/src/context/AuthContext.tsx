import React, { createContext, useContext, useState, useEffect } from 'react';
import {
  api,
  clearStoredSessionTokens,
  getStoredToken,
  setStoredSessionTokens,
} from '../services/api';
import { User } from '../types/api';

export type { User };

interface AuthContextType {
  user: User | null;
  token: string | null;
  login: (accessToken: string, user: User, refreshToken?: string | null) => void;
  logout: () => Promise<void>;
  isLoading: boolean;
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  token: null,
  login: () => {},
  logout: async () => {},
  isLoading: true,
});

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(getStoredToken());
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchUser = async () => {
      const storedToken = getStoredToken();
      if (storedToken) {
        try {
          const userData = await api.getMe();
          setUser(userData);
          setToken(getStoredToken() || storedToken);
        } catch {
          clearStoredSessionTokens();
          setToken(null);
          setUser(null);
        }
      }
      setIsLoading(false);
    };
    fetchUser();
  }, []);

  const login = (newToken: string, newUser: User, refreshToken?: string | null) => {
    setStoredSessionTokens(newToken, refreshToken);
    setToken(newToken);
    setUser(newUser);
  };

  const logout = async () => {
    await api.logout();
    setToken(null);
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, token, login, logout, isLoading }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
