from utils.report_v3 import ReportGeneratorV3


class WebReportGenerator(ReportGeneratorV3):
    """
    Web-specific report generator entry point.
    Reuses the existing report layout, then applies a web chart aligned theme.
    """

    def _generate_html(self, *args, **kwargs):
        html = super()._generate_html(*args, **kwargs)
        return self._apply_web_theme(html)

    def _apply_web_theme(self, html: str) -> str:
        themed = html.replace("<body>", '<body class="web-report-theme">', 1)

        theme_css = """
        <style>
        body.web-report-theme {
            background: #06071A !important;
            color: #E5EEF9 !important;
            max-width: 1280px !important;
            padding: 32px 40px 56px !important;
        }
        .web-report-theme h1,
        .web-report-theme h2,
        .web-report-theme h3,
        .web-report-theme h4,
        .web-report-theme h5,
        .web-report-theme h6,
        .web-report-theme strong,
        .web-report-theme .user-details .label {
            color: #F8FAFC !important;
        }
        .web-report-theme,
        .web-report-theme body,
        .web-report-theme div,
        .web-report-theme span,
        .web-report-theme .header p,
        .web-report-theme .description,
        .web-report-theme .summary-label,
        .web-report-theme .footer-warning,
        .web-report-theme .user-details span,
        .web-report-theme .tooth-details,
        .web-report-theme p,
        .web-report-theme li,
        .web-report-theme ol,
        .web-report-theme ul {
            color: #CBD5E1 !important;
        }
        .web-report-theme .label,
        .web-report-theme .legend-item span,
        .web-report-theme .footer-logo,
        .web-report-theme .odonto-label,
        .web-report-theme .summary-label,
        .web-report-theme [style*="color:#374151"],
        .web-report-theme [style*="color: #374151"],
        .web-report-theme [style*="color:#334155"],
        .web-report-theme [style*="color: #334155"],
        .web-report-theme [style*="color:#4b5563"],
        .web-report-theme [style*="color: #4b5563"],
        .web-report-theme [style*="color:#64748b"],
        .web-report-theme [style*="color: #64748b"],
        .web-report-theme [style*="color:#666"],
        .web-report-theme [style*="color: #666"] {
            color: #94A3B8 !important;
        }
        .web-report-theme .summary-val,
        .web-report-theme .footer-logo,
        .web-report-theme .ai-title,
        .web-report-theme [style*="color:#0284c7"],
        .web-report-theme [style*="color: #0284c7"],
        .web-report-theme [style*="color:#0369a1"],
        .web-report-theme [style*="color: #0369a1"],
        .web-report-theme [style*="color:#0f766e"],
        .web-report-theme [style*="color: #0f766e"] {
            color: #67E8F9 !important;
        }
        .web-report-theme .divider-thick {
            border-top: 3px solid rgba(255,255,255,0.18) !important;
        }
        .web-report-theme .divider {
            border-top: 1px solid rgba(255,255,255,0.12) !important;
        }
        .web-report-theme .pano-container,
        .web-report-theme .summary-box,
        .web-report-theme .ai-box,
        .web-report-theme .odontogram,
        .web-report-theme .tooth-card {
            background: linear-gradient(180deg, rgba(15,23,42,0.82), rgba(8,12,28,0.94)) !important;
            border: 1px solid rgba(255,255,255,0.10) !important;
            box-shadow: 0 24px 50px rgba(0,0,0,0.24) !important;
        }
        .web-report-theme .summary-title {
            color: #E5EEF9 !important;
        }
        .web-report-theme .summary-val {
            color: #22D3EE !important;
        }
        .web-report-theme .ai-box {
            background: linear-gradient(135deg, rgba(8,145,178,0.20), rgba(15,23,42,0.96)) !important;
            border-color: rgba(34,211,238,0.28) !important;
        }
        .web-report-theme .ai-title {
            color: #67E8F9 !important;
        }
        .web-report-theme .ai-content {
            color: #E0F2FE !important;
        }
        .web-report-theme .pano-img,
        .web-report-theme .crop-img {
            background: #050816 !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
        }
        .web-report-theme .legend-item {
            color: #E5EEF9 !important;
        }
        .web-report-theme .legend-item span[style*="#16a34a"] { color: #22C55E !important; }
        .web-report-theme .legend-item span[style*="#eab308"] { color: #FACC15 !important; }
        .web-report-theme .legend-item span[style*="#dc2626"] { color: #F87171 !important; }
        .web-report-theme .legend-item span[style*="#2563eb"] { color: #60A5FA !important; }
        .web-report-theme .legend-item span[style*="#9ca3af"] { color: #CBD5E1 !important; }
        .web-report-theme .odontogram h3,
        .web-report-theme .tooth-details h3,
        .web-report-theme .treatments h4,
        .web-report-theme .summary-title {
            letter-spacing: -0.02em;
        }
        .web-report-theme .odonto-tooth {
            border-radius: 12px !important;
            background: rgba(255,255,255,0.02) !important;
        }
        .web-report-theme .odonto-side {
            color: #94A3B8 !important;
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
        }
        .web-report-theme .odonto-divider {
            background: rgba(255,255,255,0.14) !important;
        }
        .web-report-theme .odonto-tooth img {
            filter: brightness(0) invert(0.92) drop-shadow(0 0 0.6px rgba(255,255,255,0.92)) !important;
            opacity: 0.9 !important;
        }
        .web-report-theme .odonto-tooth:hover {
            background: rgba(255,255,255,0.05) !important;
        }
        .web-report-theme .odonto-tooth.triage-3 {
            background: rgba(34,197,94,0.14) !important;
            border: 1px solid #22c55e !important;
        }
        .web-report-theme .odonto-tooth.triage-2 {
            background: rgba(234,179,8,0.18) !important;
            border: 1px solid #eab308 !important;
        }
        .web-report-theme .odonto-tooth.triage-1 {
            background: rgba(220,38,38,0.20) !important;
            border: 1px solid #dc2626 !important;
        }
        .web-report-theme .odonto-tooth.implant {
            background: rgba(37,99,235,0.16) !important;
            border: 1px solid #2563eb !important;
        }
        .web-report-theme .odonto-tooth.missing {
            background: rgba(209,213,219,0.12) !important;
            border: 1px dashed #94A3B8 !important;
        }
        .web-report-theme .odonto-label {
            color: #E2E8F0 !important;
            font-weight: 800 !important;
            text-shadow: 0 1px 8px rgba(2, 6, 23, 0.45) !important;
        }
        .web-report-theme .odonto-tooth.triage-3 .odonto-label { color: #4ADE80 !important; }
        .web-report-theme .odonto-tooth.triage-2 .odonto-label { color: #FACC15 !important; }
        .web-report-theme .odonto-tooth.triage-1 .odonto-label { color: #F87171 !important; }
        .web-report-theme .odonto-tooth.implant .odonto-label { color: #60A5FA !important; }
        .web-report-theme .odonto-tooth.missing .odonto-label { color: #CBD5E1 !important; }
        .web-report-theme .odonto-tooth.missing img {
            opacity: 0.26 !important;
        }
        .web-report-theme .tooth-card {
            border-radius: 24px !important;
        }
        .web-report-theme .finding-tag {
            color: #F8FAFC !important;
        }
        .web-report-theme .badge.triage-1,
        .web-report-theme .badge.overlap {
            background: #DC2626 !important;
        }
        .web-report-theme .badge.triage-2,
        .web-report-theme .badge.perio,
        .web-report-theme .badge.caries,
        .web-report-theme .badge.treatment {
            background: #D4A106 !important;
            color: #050816 !important;
        }
        .web-report-theme .badge.triage-3,
        .web-report-theme .badge.pbl-1,
        .web-report-theme .badge.pbl-2,
        .web-report-theme .badge.info,
        .web-report-theme .badge.success {
            background: #16A34A !important;
        }
        .web-report-theme .badge.implant {
            background: #2563EB !important;
        }
        .web-report-theme .footer {
            border-top: 1px solid rgba(255,255,255,0.12) !important;
        }
        .web-report-theme a,
        .web-report-theme a:visited {
            color: #67E8F9 !important;
        }
        .web-report-theme [style*="background-color: #fffbeb"],
        .web-report-theme [style*="background-color:#fffbeb"] {
            background: linear-gradient(135deg, rgba(146,64,14,0.24), rgba(15,23,42,0.96)) !important;
            border: 1px solid rgba(251,191,36,0.22) !important;
            border-left: 4px solid #F59E0B !important;
            box-shadow: 0 18px 36px rgba(0,0,0,0.18) !important;
        }
        .web-report-theme [style*="color: #d97706"],
        .web-report-theme [style*="color:#d97706"] {
            color: #FBBF24 !important;
        }
        </style>
        """

        replacements = {
            "background: #fff;": "background: #06071A;",
            "color: #333;": "color: #E5EEF9;",
            "background: #fafafa;": "background: linear-gradient(180deg, rgba(15,23,42,0.82), rgba(8,12,28,0.94));",
            "background: #f0f9ff;": "background: linear-gradient(180deg, rgba(8,145,178,0.14), rgba(15,23,42,0.94));",
            "border: 1px solid #bae6fd;": "border: 1px solid rgba(34,211,238,0.28);",
            "background-color: #fffbeb;": "background: linear-gradient(135deg, rgba(146,64,14,0.24), rgba(15,23,42,0.96));",
            "border-left: 5px solid #d97706;": "border-left: 4px solid #F59E0B;",
            "background: #f5f5f5;": "background: #050816;",
            "border-top: 1px solid #ddd;": "border-top: 1px solid rgba(255,255,255,0.12);",
            "background: #fff;": "background: linear-gradient(180deg, rgba(15,23,42,0.82), rgba(8,12,28,0.94));",
            "border: 2px solid #e5e7eb;": "border: 1px solid rgba(255,255,255,0.10);",
            "background: #000;": "background: #22D3EE;",
            "color: #666;": "color: #94A3B8;",
            "color: #4b5563;": "color: #CBD5E1;",
            "color: #d97706;": "color: #FBBF24;",
        }

        for source, target in replacements.items():
            themed = themed.replace(source, target)

        if "</head>" in themed:
            themed = themed.replace("</head>", f"{theme_css}</head>", 1)
        else:
            themed = theme_css + themed
        return themed
