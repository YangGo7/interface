import { useEffect, useState, type ReactNode } from 'react';
import { Link, useParams } from 'react-router-dom';
import { fetchWebReportSession, type WebReportSessionResponse } from '../lib/webReportApi';
import logo from '../assets/O3_logo_only.png';

export function WebReportPage() {
  const { sessionId = '' } = useParams();
  const [session, setSession] = useState<WebReportSessionResponse['session'] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const isKorean = session?.language === 'Korean';

  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;

    const load = async () => {
      try {
        const data = await fetchWebReportSession(sessionId);
        if (!cancelled) {
          setSession(data.session || null);
          setError(data.error || null);
        }
      } catch (err: any) {
        if (!cancelled) {
          setError(err?.message || 'Failed to load report session');
        }
      }
    };

    load();
    const timer = window.setInterval(load, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [sessionId]);

  useEffect(() => {
    if (!sessionId || !session?.report?.html_url) return;
    if (session.status === 'completed' || session.status === 'finalized') {
      window.location.replace(`/api/web_report/session/${sessionId}/report`);
    }
  }, [sessionId, session?.report?.html_url, session?.status]);

  if (!sessionId) {
    return <SimpleState title="Invalid session" detail="No session id was provided." />;
  }

  if (!session) {
    return <SimpleState title="Loading report" detail="Connecting to the report session..." />;
  }

  if ((session.status !== 'completed' && session.status !== 'finalized') || !session.report?.html_url) {
    return (
      <SimpleState
        title={isKorean ? '리포트 준비 중' : 'Report not ready'}
        detail={error || (isKorean ? `현재 상태: ${session.status}` : `Current status: ${session.status}`)}
        action={<Link to={`/chart/${sessionId}`} className="inline-flex rounded-full border border-slate-700 px-4 py-2 text-sm text-slate-200 hover:bg-slate-800">{isKorean ? '차트로 돌아가기' : 'Back to chart'}</Link>}
      />
    );
  }

  const reportUrl = `/api/web_report/session/${sessionId}/report`;
  const pdfUrl = `/api/web_report/session/${sessionId}/report/pdf`;
  const hasPdf = Boolean(session.report?.pdf_path);

  return (
    <div className="min-h-screen bg-slate-950 px-6 py-12 text-slate-100">
      <div className="mx-auto max-w-2xl rounded-3xl border border-slate-800 bg-slate-900 p-8 shadow-2xl">
        <div className="flex items-center gap-3">
          <img src={logo} alt="Logo" className="h-8 w-auto object-contain" />
          <span className="font-bold text-lg tracking-tight">{isKorean ? '전체 리포트 여는 중' : 'Opening full report'}</span>
        </div>
        <p className="mt-3 text-sm text-slate-300">
          {isKorean
            ? `세션 ${sessionId.slice(0, 8)} 리포트 HTML로 이동합니다.`
            : `Redirecting to the report HTML for session ${sessionId.slice(0, 8)}.`}
        </p>
        <div className="mt-6 flex items-center gap-3">
          <a
            href={reportUrl}
            target="_self"
            rel="noreferrer"
            className="inline-flex rounded-full bg-slate-100 px-4 py-2 text-sm font-medium text-slate-900"
          >
            {isKorean ? '지금 열기' : 'Open now'}
          </a>
          {hasPdf ? (
            <a
              href={pdfUrl}
              target="_blank"
              rel="noreferrer"
              className="inline-flex rounded-full border border-slate-700 px-4 py-2 text-sm font-medium text-slate-100"
            >
              PDF
            </a>
          ) : null}
          <Link
            to={`/chart/${sessionId}`}
            className="inline-flex rounded-full border border-slate-700 px-4 py-2 text-sm font-medium text-slate-100"
          >
            {isKorean ? '차트로 돌아가기' : 'Back to Chart'}
          </Link>
        </div>
      </div>
    </div>
  );
}

function SimpleState({
  title,
  detail,
  action,
}: {
  title: string;
  detail: string;
  action?: ReactNode;
}) {
  return (
    <div className="min-h-screen bg-slate-950 px-6 py-12 text-slate-100">
      <div className="mx-auto max-w-2xl rounded-3xl border border-slate-800 bg-slate-900 p-8 shadow-2xl">
        <h1 className="text-2xl font-semibold">{title}</h1>
        <p className="mt-3 text-sm text-slate-300">{detail}</p>
        {action ? <div className="mt-6">{action}</div> : null}
      </div>
    </div>
  );
}
