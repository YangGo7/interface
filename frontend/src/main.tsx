import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import './index.css';
import App from './App';
import { ChartPage } from './pages/ChartPage';
import { TestPage } from './pages/TestPage';
import { ReportPage } from './pages/ReportPage';
import { CornerstonePage } from './pages/CornerstonePage';
import { MprTestPage } from './pages/MprTestPage';

type RouteErrorBoundaryProps = {
  children: React.ReactNode;
  routeLabel: string;
};

type RouteErrorBoundaryState = {
  hasError: boolean;
  errorMessage: string;
};

class RouteErrorBoundary extends React.Component<RouteErrorBoundaryProps, RouteErrorBoundaryState> {
  state: RouteErrorBoundaryState = {
    hasError: false,
    errorMessage: '',
  };

  static getDerivedStateFromError(error: Error): RouteErrorBoundaryState {
    return {
      hasError: true,
      errorMessage: error?.message || 'Unknown error',
    };
  }

  componentDidCatch(error: Error) {
    console.error(`Route "${this.props.routeLabel}" crashed`, error);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-slate-950 px-6 py-12 text-slate-100">
          <div className="mx-auto max-w-2xl rounded-3xl border border-red-500/30 bg-slate-900 p-8 shadow-2xl">
            <h1 className="text-2xl font-semibold">Page failed to load</h1>
            <p className="mt-3 text-sm text-slate-300">Route: {this.props.routeLabel}</p>
            <p className="mt-2 text-sm text-red-300">{this.state.errorMessage}</p>
            <a
              href="/"
              className="mt-6 inline-flex rounded-full bg-slate-100 px-4 py-2 text-sm font-medium text-slate-900"
            >
              Return home
            </a>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<RouteErrorBoundary routeLabel="/"><App /></RouteErrorBoundary>} />
        <Route path="/chart" element={<RouteErrorBoundary routeLabel="/chart"><ChartPage /></RouteErrorBoundary>} />
        <Route path="/test" element={<RouteErrorBoundary routeLabel="/test"><TestPage /></RouteErrorBoundary>} />
        <Route path="/report" element={<RouteErrorBoundary routeLabel="/report"><ReportPage /></RouteErrorBoundary>} />
        <Route path="/cornerstone_page" element={<RouteErrorBoundary routeLabel="/cornerstone_page"><CornerstonePage /></RouteErrorBoundary>} />
        <Route path="/mpr_test" element={<RouteErrorBoundary routeLabel="/mpr_test"><MprTestPage /></RouteErrorBoundary>} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);
