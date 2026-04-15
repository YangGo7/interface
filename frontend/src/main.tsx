import React, { Suspense, lazy } from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import './index.css';
import App from './App';

const ChartPage = lazy(() => import('./pages/ChartPage').then((module) => ({ default: module.ChartPage })));
const TestPage = lazy(() => import('./pages/TestPage').then((module) => ({ default: module.TestPage })));
const ReportPage = lazy(() => import('./pages/ReportPage').then((module) => ({ default: module.ReportPage })));
const CornerstonePage = lazy(() => import('./pages/CornerstonePage').then((module) => ({ default: module.CornerstonePage })));
const MprTestPage = lazy(() => import('./pages/MprTestPage').then((module) => ({ default: module.MprTestPage })));
const WebChartPage = lazy(() => import('./pages/WebChartPage').then((module) => ({ default: module.WebChartPage })));
const WebReportPage = lazy(() => import('./pages/WebReportPage').then((module) => ({ default: module.WebReportPage })));
const RenewPage = lazy(() => import('./pages/RenewPage').then((module) => ({ default: module.RenewPage })));
const UploadPage = lazy(() => import('./pages/UploadPage'));
const FolderLeaderPage = lazy(() => import('./pages/FolderLeaderPage'));
const FolderLeaderVer2Page = lazy(() => import('./pages/FolderLeaderVer2Page'));

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

function RouteLoadingFallback() {
  return (
    <div className="min-h-screen bg-slate-950 px-6 py-12 text-slate-100">
      <div className="mx-auto max-w-2xl rounded-3xl border border-slate-800 bg-slate-900 p-8 shadow-2xl">
        <h1 className="text-2xl font-semibold">Loading page</h1>
        <p className="mt-3 text-sm text-slate-300">Please wait while the route bundle loads.</p>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Suspense fallback={<RouteLoadingFallback />}>
        <Routes>
          <Route path="/" element={<RouteErrorBoundary routeLabel="/"><App /></RouteErrorBoundary>} />
          <Route path="/folder-leader" element={<RouteErrorBoundary routeLabel="/folder-leader"><FolderLeaderPage /></RouteErrorBoundary>} />
          <Route path="/folder_leader_ver_2" element={<RouteErrorBoundary routeLabel="/folder_leader_ver_2"><FolderLeaderVer2Page /></RouteErrorBoundary>} />
          <Route path="/upload" element={<RouteErrorBoundary routeLabel="/upload"><UploadPage /></RouteErrorBoundary>} />
          <Route path="/chart" element={<RouteErrorBoundary routeLabel="/chart"><ChartPage /></RouteErrorBoundary>} />
          <Route path="/chart/:sessionId" element={<RouteErrorBoundary routeLabel="/chart/:sessionId"><WebChartPage /></RouteErrorBoundary>} />
          <Route path="/renew" element={<RouteErrorBoundary routeLabel="/renew"><RenewPage /></RouteErrorBoundary>} />
          <Route path="/test" element={<RouteErrorBoundary routeLabel="/test"><TestPage /></RouteErrorBoundary>} />
          <Route path="/report" element={<RouteErrorBoundary routeLabel="/report"><ReportPage /></RouteErrorBoundary>} />
          <Route path="/report/:sessionId" element={<RouteErrorBoundary routeLabel="/report/:sessionId"><WebReportPage /></RouteErrorBoundary>} />
          <Route path="/cornerstone_page" element={<RouteErrorBoundary routeLabel="/cornerstone_page"><CornerstonePage /></RouteErrorBoundary>} />
          <Route path="/mpr_test" element={<RouteErrorBoundary routeLabel="/mpr_test"><MprTestPage /></RouteErrorBoundary>} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  </React.StrictMode>
);
