import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import './index.css';
import App from './App';
import { ChartPage } from './pages/ChartPage';
import { TestPage } from './pages/TestPage';
import { ReportPage } from './pages/ReportPage';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/chart" element={<ChartPage />} />
        <Route path="/test" element={<TestPage />} />
        <Route path="/report" element={<ReportPage />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);
