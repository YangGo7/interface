from utils.report_v3 import ReportGeneratorV3


class WebReportGenerator(ReportGeneratorV3):
    """
    Web-specific report generator entry point.
    Kept separate from the GPTs v2 flow so web report evolution can diverge safely.
    """

    pass
