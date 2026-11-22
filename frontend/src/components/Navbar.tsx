import React, { useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import {
  Leaf,
  LayoutDashboard,
  ScanLine,
  History,
  Shield,
  LogOut,
  LogIn,
  Menu,
  X,
} from 'lucide-react';

export const Navbar: React.FC = () => {
  const { user, logout } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const [mobileOpen, setMobileOpen] = useState(false);

  const isActive = (path: string) => location.pathname === path;

  const navLinkClass = (path: string) =>
    `flex items-center px-3 py-2 rounded-xl text-sm font-medium transition ${
      isActive(path)
        ? 'bg-brand-50 text-brand-700 font-semibold border border-brand-200/70'
        : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
    }`;

  const handleLogout = async () => {
    setMobileOpen(false);
    await logout();
    navigate('/auth');
  };

  return (
    <nav className="bg-white/95 backdrop-blur border-b border-slate-200 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="flex items-center space-x-2.5" onClick={() => setMobileOpen(false)}>
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-brand-600 to-brand-900 flex items-center justify-center text-white shadow-sm">
                <Leaf className="w-5 h-5" />
              </div>
              <div>
                <span className="text-lg font-extrabold text-slate-900 tracking-tight">
                  CottonGuard<span className="text-brand-600"> AI</span>
                </span>
                <span className="hidden lg:inline-block ml-2 text-[11px] font-semibold px-2 py-0.5 bg-brand-50 text-brand-700 rounded-full border border-brand-200">
                  Precision Foliar Diagnostics
                </span>
              </div>
            </Link>

            {user && (
              <div className="hidden md:flex ml-8 space-x-1.5">
                <Link to="/dashboard" className={navLinkClass('/dashboard')}>
                  <LayoutDashboard className="w-4 h-4 mr-1.5" /> Dashboard
                </Link>
                <Link to="/analyze" className={navLinkClass('/analyze')}>
                  <ScanLine className="w-4 h-4 mr-1.5" /> Analyze Leaf
                </Link>
                <Link to="/history" className={navLinkClass('/history')}>
                  <History className="w-4 h-4 mr-1.5" /> History
                </Link>
                {user.role === 'admin' && (
                  <Link to="/admin" className={navLinkClass('/admin')}>
                    <Shield className="w-4 h-4 mr-1.5" /> Admin
                  </Link>
                )}
              </div>
            )}
          </div>

          <div className="flex items-center space-x-3">
            {user ? (
              <div className="hidden md:flex items-center space-x-3">
                <div className="text-right">
                  <div className="text-xs font-bold text-slate-800 leading-none">{user.username}</div>
                  <div className="text-[11px] uppercase tracking-wider text-brand-700 font-semibold mt-0.5">
                    {user.role}
                  </div>
                </div>
                <button
                  type="button"
                  onClick={handleLogout}
                  className="flex items-center px-3 py-1.5 rounded-xl text-sm text-slate-600 hover:text-red-600 hover:bg-red-50 transition border border-slate-200"
                >
                  <LogOut className="w-4 h-4 mr-1.5" /> Logout
                </button>
              </div>
            ) : (
              <Link
                to="/auth"
                className="flex items-center px-4 py-2 rounded-xl text-sm font-semibold bg-brand-600 hover:bg-brand-700 text-white shadow-sm transition"
              >
                <LogIn className="w-4 h-4 mr-1.5" /> Sign In
              </Link>
            )}

            <button
              type="button"
              onClick={() => setMobileOpen(!mobileOpen)}
              className="md:hidden p-2 rounded-lg text-slate-600 hover:bg-slate-100"
              aria-label="Toggle Menu"
            >
              {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>

        {/* Responsive Mobile Menu */}
        {mobileOpen && (
          <div className="md:hidden py-3 border-t border-slate-100 space-y-1">
            {user ? (
              <>
                <Link to="/dashboard" onClick={() => setMobileOpen(false)} className={navLinkClass('/dashboard')}>
                  <LayoutDashboard className="w-4 h-4 mr-2" /> Dashboard
                </Link>
                <Link to="/analyze" onClick={() => setMobileOpen(false)} className={navLinkClass('/analyze')}>
                  <ScanLine className="w-4 h-4 mr-2" /> Analyze Leaf
                </Link>
                <Link to="/history" onClick={() => setMobileOpen(false)} className={navLinkClass('/history')}>
                  <History className="w-4 h-4 mr-2" /> History
                </Link>
                {user.role === 'admin' && (
                  <Link to="/admin" onClick={() => setMobileOpen(false)} className={navLinkClass('/admin')}>
                    <Shield className="w-4 h-4 mr-2" /> Admin Console
                  </Link>
                )}
                <button
                  type="button"
                  onClick={handleLogout}
                  className="w-full flex items-center px-3 py-2 rounded-xl text-sm font-medium text-red-600 hover:bg-red-50"
                >
                  <LogOut className="w-4 h-4 mr-2" /> Logout ({user.username})
                </button>
              </>
            ) : (
              <Link to="/auth" onClick={() => setMobileOpen(false)} className={navLinkClass('/auth')}>
                <LogIn className="w-4 h-4 mr-2" /> Sign In / Register
              </Link>
            )}
          </div>
        )}
      </div>
    </nav>
  );
};
