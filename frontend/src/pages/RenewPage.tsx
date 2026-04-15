import { useEffect, useState } from 'react';
import O3Logo from '../assets/O3_logo_only.png';

const DESIGN_WIDTH = 1920;
const DESIGN_HEIGHT = 1080;

const wp = (value: number) => `${value}px`;
const hp = (value: number) => `${value}px`;
const scalePx = (value: number) => `${value}px`;
const relativePercent = (value: number, total: number) => `${(value / total) * 100}%`;

const assetPath = (relativePath: string) => encodeURI(`/imgs/${relativePath}`);

const reportButtonIcons = {
  inactive: assetPath('메인-비활성화 아이콘/report버튼 (94 94).png'),
  active: assetPath('메인-클릭/report버튼 (94 94).png'),
};

const railIcons = {
  studies: assetPath('메인-클릭/좌측 bar 아이콘 (51 57)/자산 425@4x.png'),
  captures: assetPath('메인-비활성화 아이콘/좌측 bar 아이콘 (51 57)/자산 584.png'),
  report: assetPath('메인-비활성화 아이콘/report버튼 (94 94).png'),
};

const toolbarIcons = [
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 20@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 21@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 22@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 23@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 24@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 26@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 31@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 30@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 25@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 29@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 28@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 27@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 35@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 34@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 33@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 32@4x.png'),
];

const displayRailIcons = {
  studies: assetPath('mian_deactive/left_bar_deactive (51 57)/자산 427@4x.png'),
  captures: assetPath('mian_deactive/left_bar_deactive (51 57)/자산 584.png'),
};

const displayToolbarIcons = [
  assetPath('mian_deactive/tools_deactive(36 36)/자산 20@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 21@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 22@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 23@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 24@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 26@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 31@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 30@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 25@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 29@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 28@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 27@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 35@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 34@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 33@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 32@4x.png'),
];

const displayReportButtonIcons = {
  inactive: assetPath('mian_deactive/report_deactive.png'),
  active: assetPath('main_active/report_active (94 94).png'),
};

const legendItems = [
  { label: 'Urgent Priority', color: '#FF0037', top: 834 },
  { label: 'Treatment Required', color: '#FCFF2A', top: 867 },
  { label: 'Implant', color: '#003DFF', top: 900 },
  { label: 'Missing Tooth', color: '#3F3F3F', top: 933 },
  { label: 'Healthy Tooth', color: '#FFFFFF', top: 966 },
] as const;

const upperSizes = ['23 79', '20 75', '22 82', '25 74', '25 67', '38 58', '36 54', '35 49'];
const lowerSizes = ['18 71', '17 70', '22 78', '25 75', '26 76', '39 68', '38 65', '41 63'];
const upperLeftOrders = [8, 7, 6, 5, 4, 3, 2, 1] as const;
const upperRightOrders = [1, 2, 3, 4, 5, 6, 7, 8] as const;
const lowerLeftOrders = [8, 7, 6, 5, 4, 3, 2, 1] as const;
const lowerRightOrders = [1, 2, 3, 4, 5, 6, 7, 8] as const;
const upperTeeth = [18, 17, 16, 15, 14, 13, 12, 11] as const;
const upperRightTeeth = [21, 22, 23, 24, 25, 26, 27, 28] as const;
const lowerTeeth = [48, 47, 46, 45, 44, 43, 42, 41] as const;
const lowerRightTeeth = [31, 32, 33, 34, 35, 36, 37, 38] as const;
const outerToInnerOffsets = [390.5, 332.5, 274.5, 216.5, 166.5, 116.5, 66.5, 22.5] as const;
const innerToOuterOffsets = [22.5, 66.5, 116.5, 166.5, 216.5, 274.5, 332.5, 390.5] as const;
const upperBaseline = 935;
const lowerTop = 953;
const chartSectionTop = 804;
const chartHeaderHeight = 18;
const chartContentTop = chartSectionTop + chartHeaderHeight;
const chartContentHeight = 246;
const chartLegendLeft = 222;
const chartLegendTop = 844;
const chartLegendHeight = 151;
const RAIL_ICON_WIDTH = 61;
const RAIL_ICON_HEIGHT = 68;
const TOOL_ICON_SIZE = 43;

function getToothAsset(tooth: number) {
  const index = (tooth % 10) - 1;
  const upper = tooth >= 11 && tooth <= 28;
  const prefix = upper ? 'U' : 'L';
  const size = upper ? upperSizes[index] : lowerSizes[index];
  return assetPath(`치아/건강치(ffffff)/${prefix}-${index + 1} (${size})_4.png`);
}

function ToolIcon({ icon, left, top }: { icon: string; left: number; top: number }) {
  return (
    <div
      style={{
        width: wp(TOOL_ICON_SIZE),
        height: hp(TOOL_ICON_SIZE),
        left: wp(left),
        top: hp(top),
        position: 'absolute',
        border: `${scalePx(1)} solid black`,
        background: '#8D8D8D',
      }}
    >
      <img
        src={icon}
        alt=""
        draggable={false}
        style={{ width: '100%', height: '100%', position: 'absolute', left: 0, top: 0 }}
      />
    </div>
  );
}

function ToothImage({
  tooth,
  left,
  top,
  width,
  height,
  rotate = false,
}: {
  tooth: number;
  left: number;
  top: number;
  width: number;
  height: number;
  rotate?: boolean;
}) {
  return (
    <img
      src={getToothAsset(tooth)}
      alt=""
      draggable={false}
      style={{
        width: wp(width),
        height: hp(height),
        left: wp(left),
        top: hp(top),
        position: 'absolute',
        transform: rotate ? `translate(${width}px, ${height}px) rotate(180deg)` : undefined,
        transformOrigin: rotate ? 'top left' : undefined,
      }}
    />
  );
}

function getHealthToothAsset(arch: 'U' | 'L', order: number) {
  const size = arch === 'U' ? upperSizes[order - 1] : lowerSizes[order - 1];
  return assetPath(`teeth/health(ffffff)/${arch}-${order} (${size})_4.png`);
}

function ToothSlotImage({
  arch,
  order,
  left,
  top,
  width,
  height,
  flipX = false,
}: {
  arch: 'U' | 'L';
  order: number;
  left: number;
  top: number;
  width: number;
  height: number;
  flipX?: boolean;
}) {
  return (
    <img
      src={getHealthToothAsset(arch, order)}
      alt=""
      draggable={false}
      style={{
        width: wp(width),
        height: hp(height),
        left: wp(left),
        top: hp(top),
        position: 'absolute',
        transform: flipX ? `translate(${width}px, 0) scaleX(-1)` : undefined,
        transformOrigin: flipX ? 'top left' : undefined,
      }}
    />
  );
}

function useViewportSize() {
  const [viewport, setViewport] = useState(() => ({
    width: typeof window === 'undefined' ? DESIGN_WIDTH : window.innerWidth,
    height: typeof window === 'undefined' ? DESIGN_HEIGHT : window.innerHeight,
  }));

  useEffect(() => {
    const updateViewport = () => {
      setViewport({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    };

    updateViewport();
    window.addEventListener('resize', updateViewport);
    return () => window.removeEventListener('resize', updateViewport);
  }, []);

  return viewport;
}

export function RenewPage() {
  const [isReportActive, setIsReportActive] = useState(false);
  const viewport = useViewportSize();
  const scale = Math.min(viewport.width / DESIGN_WIDTH, viewport.height / DESIGN_HEIGHT);
  const stageWidth = viewport.width;
  const stageHeight = DESIGN_HEIGHT * scale;
  const designCanvasWidth = stageWidth / scale;
  const viewerExtraWidth = Math.max(0, designCanvasWidth - DESIGN_WIDTH);
  const viewerLeft = 234;
  const topBarLeft = 241;
  const viewerWidth = 1676 + viewerExtraWidth;
  const topBarWidth = 1666 + viewerExtraWidth;
  const rightEdge = viewerLeft + viewerWidth;
  const chartSectionLeft = 236;
  const chartSectionWidth = rightEdge - chartSectionLeft;
  const chartLegendDividerX = chartSectionLeft + chartSectionWidth * 0.15;
  const chartLegendWidth = chartLegendDividerX - chartLegendLeft - 28;
  const reportLeft = rightEdge - 106;
  const odontoFrameLeft = chartLegendDividerX + 36;
  const odontoFrameRight = reportLeft - 58;
  const chartOdontoCenterX = (odontoFrameLeft + odontoFrameRight) / 2;
  const toothSlotCentersLeft = outerToInnerOffsets.map((offset) => chartOdontoCenterX - offset);
  const toothSlotCentersRight = innerToOuterOffsets.map((offset) => chartOdontoCenterX + offset);
  const maxUpperHeight = Math.max(...upperSizes.map((size) => Number(size.split(' ')[1])));
  const maxLowerHeight = Math.max(...lowerSizes.map((size) => Number(size.split(' ')[1])));
  const chartOdontoVerticalPadding = 10;
  const chartOdontoFrameTop = upperBaseline - maxUpperHeight - chartOdontoVerticalPadding;
  const chartOdontoFrameBottom = lowerTop + maxLowerHeight + chartOdontoVerticalPadding;
  const chartOdontoLineLeft = toothSlotCentersLeft[0] - 24;
  const chartOdontoLineRight = toothSlotCentersRight[toothSlotCentersRight.length - 1] + 24;
  const chartOdontoLineWidth = chartOdontoLineRight - chartOdontoLineLeft;
  const chartOdontoLineY = Math.round((upperBaseline + lowerTop) / 2);
  const chartOdontoVerticalTop = chartOdontoFrameTop;
  const chartOdontoVerticalHeight = chartOdontoFrameBottom - chartOdontoFrameTop;
  const viewLabelTop = 75;
  const measureLabelTop = 195;
  const outputLabelTop = 315;
  const sectionArrowOffsetY = 6;
  const reportTop = chartContentTop + 130;
  const fdiLeft = rightEdge - 82;
  const lLabelLeft = rightEdge - 29;

  return (
    <div
      style={{
        width: '100vw',
        height: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#000000',
        overflow: 'hidden',
        fontFamily: '"Noto Sans", "Noto Sans KR", sans-serif',
      }}
    >
      <div
        style={{
          width: `${stageWidth}px`,
          height: `${stageHeight}px`,
          position: 'relative',
          overflow: 'hidden',
          background: '#414950',
          boxShadow: 'inset 1px 0 0 rgba(0, 0, 0, 0.18), inset -1px 0 0 rgba(0, 0, 0, 0.18)',
        }}
      >
        <div
          style={{
            width: wp(designCanvasWidth),
            height: hp(DESIGN_HEIGHT),
            left: 0,
            top: 0,
            position: 'absolute',
            background: '#414950',
            transform: `scale(${scale})`,
            transformOrigin: 'top left',
          }}
        >
          <div style={{ width: wp(designCanvasWidth), height: hp(DESIGN_HEIGHT), left: 0, top: 0, position: 'absolute', background: '#414950' }} />

          <img
            src={O3Logo}
            alt="O3"
            draggable={false}
            style={{
              width: wp(144),
              height: hp(38),
              left: wp(-20),
              top: hp(10),
              position: 'absolute',
              objectFit: 'contain',
              zIndex: 12,
            }}
          />
          <div
            style={{
              left: wp(88),
              top: hp(12),
              position: 'absolute',
              color: 'white',
              fontSize: scalePx(20),
              fontWeight: 700,
              letterSpacing: '0.08em',
              zIndex: 12,
            }}
          >
            SATURN
          </div>

          <div style={{ width: wp(topBarWidth), height: hp(18), left: wp(topBarLeft), top: hp(49), position: 'absolute', background: '#5C5C5C' }} />
          <div style={{ width: wp(viewerWidth), height: hp(1018), left: wp(viewerLeft), top: hp(50), position: 'absolute', background: 'black' }} />
          <div style={{ width: wp(viewerWidth), height: hp(18), left: wp(viewerLeft), top: hp(804), position: 'absolute', background: '#5C5C5C' }} />
          <div style={{ width: wp(viewerWidth), height: hp(755), left: wp(viewerLeft), top: hp(49), position: 'absolute', border: `${scalePx(1)} solid #4C4C4C`, pointerEvents: 'none' }} />
          <div style={{ width: wp(viewerWidth), height: hp(264), left: wp(viewerLeft), top: hp(804), position: 'absolute', border: `${scalePx(1)} solid #4C4C4C`, pointerEvents: 'none' }} />
          <div style={{ width: scalePx(1), height: hp(1019), left: wp(viewerLeft), top: hp(49), position: 'absolute', background: '#4C4C4C' }} />

        <div style={{ width: wp(70), height: hp(1019), left: wp(12), top: hp(49), position: 'absolute', background: '#2D2D2D' }} />
        <div style={{ width: wp(70), height: hp(1019), left: wp(12), top: hp(49), position: 'absolute', border: `${scalePx(1)} solid #4C4C4C`, pointerEvents: 'none' }} />
        {/* 먼 세로선이지 */} 
        {/*<div style={{ width: scalePx(1), height: hp(1019), left: wp(48), top: hp(49), position: 'absolute', background: '#3F3F3F' }} />  
        <div style={{ width: scalePx(1), height: hp(1019), left: wp(68), top: hp(49), position: 'absolute', background: '#5C5C5C' }} /> */}
        <div style={{ width: wp(RAIL_ICON_WIDTH), height: hp(RAIL_ICON_HEIGHT), left: wp(16), top: hp(52), position: 'absolute', background: '#2D2D2D' }} />
        <div style={{ width: wp(RAIL_ICON_WIDTH), height: hp(RAIL_ICON_HEIGHT), left: wp(16), top: hp(122), position: 'absolute', background: '#2D2D2D' }} />
        <img style={{ width: wp(RAIL_ICON_WIDTH), height: hp(RAIL_ICON_HEIGHT), left: wp(16), top: hp(52), position: 'absolute' }} src={displayRailIcons.studies} alt="" />
        <img style={{ width: wp(RAIL_ICON_WIDTH), height: hp(RAIL_ICON_HEIGHT), left: wp(16), top: hp(122), position: 'absolute' }} src={displayRailIcons.captures} alt="" />

        <div style={{ width: wp(154), height: hp(1019), left: wp(82), top: hp(49), position: 'absolute', background: '#2D2D2D' }} />
        <div style={{ width: wp(154), height: hp(1019), left: wp(82), top: hp(49), position: 'absolute', border: `${scalePx(1)} solid #4C4C4C`, pointerEvents: 'none' }} />
        <div style={{ width: wp(148), height: hp(334), left: wp(86), top: hp(51), position: 'absolute', background: '#333333' }} />
        <div style={{ width: wp(148), height: hp(84), left: wp(86), top: hp(389), position: 'absolute', background: '#333333' }} />
        <div style={{ width: wp(140), height: hp(116), left: wp(88), top: hp(69), position: 'absolute', background: '#414141' }} />
        <div style={{ width: wp(140), height: hp(118), left: wp(88), top: hp(189), position: 'absolute', background: '#414141' }} />
        <div style={{ width: wp(140), height: hp(72), left: wp(88), top: hp(309), position: 'absolute', background: '#414141' }} />
        <div style={{ width: wp(140), height: hp(52), left: wp(88), top: hp(411), position: 'absolute', background: '#414141' }} />

        <div style={{ left: wp(109), top: hp(50), position: 'absolute', color: 'white', fontSize: scalePx(14), fontWeight: 700 }}>General</div>
        <div style={{ left: wp(109), top: hp(viewLabelTop), position: 'absolute', color: 'white', fontSize: scalePx(10), fontWeight: 700 }}>View</div>
        <div style={{ left: wp(109), top: hp(measureLabelTop), position: 'absolute', color: 'white', fontSize: scalePx(13), fontWeight: 700 }}>Measure</div>
        <div style={{ left: wp(109), top: hp(outputLabelTop), position: 'absolute', color: 'white', fontSize: scalePx(13), fontWeight: 700 }}>Output</div>
        <div style={{ left: wp(109), top: hp(390), position: 'absolute', color: 'white', fontSize: scalePx(14), fontWeight: 700 }}>Task</div>

        <div style={{ width: scalePx(7), height: scalePx(7), left: wp(96), top: hp(57), position: 'absolute', background: '#D9D9D9' }} />
        <div style={{ width: scalePx(7), height: scalePx(7), left: wp(96), top: hp(397), position: 'absolute', background: '#D9D9D9' }} />
        <div
          style={{
            width: 0,
            height: 0,
            left: wp(96),
            top: hp(viewLabelTop + sectionArrowOffsetY),
            position: 'absolute',
            borderLeft: `${scalePx(4)} solid transparent`,
            borderRight: `${scalePx(4)} solid transparent`,
            borderTop: `${scalePx(8)} solid #D9D9D9`,
          }}
        />
        <div
          style={{
            width: 0,
            height: 0,
            left: wp(96),
            top: hp(measureLabelTop + sectionArrowOffsetY),
            position: 'absolute',
            borderLeft: `${scalePx(4)} solid transparent`,
            borderRight: `${scalePx(4)} solid transparent`,
            borderTop: `${scalePx(8)} solid #D9D9D9`,
          }}
        />
        <div
          style={{
            width: 0,
            height: 0,
            left: wp(96),
            top: hp(outputLabelTop + sectionArrowOffsetY),
            position: 'absolute',
            borderLeft: `${scalePx(4)} solid transparent`,
            borderRight: `${scalePx(4)} solid transparent`,
            borderTop: `${scalePx(8)} solid #D9D9D9`,
          }}
        />

        {[
          { x: 91.5, y: 96, icon: displayToolbarIcons[0] },
          { x: 136.5, y: 96, icon: displayToolbarIcons[1] },
          { x: 181.5, y: 96, icon: displayToolbarIcons[2] },
          { x: 91.5, y: 141, icon: displayToolbarIcons[3] },
          { x: 136.5, y: 141, icon: displayToolbarIcons[4] },
          { x: 181.5, y: 141, icon: displayToolbarIcons[5] },
          { x: 91.5, y: 216, icon: displayToolbarIcons[6] },
          { x: 136.5, y: 216, icon: displayToolbarIcons[7] },
          { x: 181.5, y: 216, icon: displayToolbarIcons[8] },
          { x: 91.5, y: 261, icon: displayToolbarIcons[9] },
          { x: 136.5, y: 261, icon: displayToolbarIcons[10] },
          { x: 181.5, y: 261, icon: displayToolbarIcons[11] },
          { x: 92, y: 336, icon: displayToolbarIcons[12] },
          { x: 139, y: 336, icon: displayToolbarIcons[13] },
          { x: 92, y: 416, icon: displayToolbarIcons[14] },
          { x: 139, y: 416, icon: displayToolbarIcons[15] },
        ].map((item, index) => (
          <ToolIcon key={index} icon={item.icon} left={item.x} top={item.y} />
        ))}

        <div style={{ left: wp(251), top: hp(49), position: 'absolute', color: 'white', fontSize: scalePx(13), fontWeight: 700 }}>Panorama</div>
        <div style={{ width: scalePx(7), height: scalePx(7), left: wp(240), top: hp(55), position: 'absolute', background: '#D9D9D9' }} />
        <div style={{ left: wp(251), top: hp(803), position: 'absolute', color: 'white', fontSize: scalePx(13), fontWeight: 700 }}>Dental Chart</div>
        <div style={{ width: scalePx(7), height: scalePx(7), left: wp(240), top: hp(809), position: 'absolute', background: '#D9D9D9' }} />

        <div style={{ left: wp(fdiLeft), top: hp(845), position: 'absolute', fontWeight: 700 }}>
          <span style={{ color: 'white', fontSize: scalePx(14) }}>FDI</span>
          <span style={{ color: 'white', fontSize: scalePx(13) }}> </span>
          <span style={{ color: '#9C9C9C', fontSize: scalePx(13) }}>/ Univ</span>
        </div>

        <div style={{ left: wp(251), top: hp(415), position: 'absolute', color: '#D39C00', fontSize: scalePx(14), fontWeight: 700, zIndex: 14 }}>R</div>
        <div style={{ left: wp(lLabelLeft), top: hp(415), position: 'absolute', color: '#D39C00', fontSize: scalePx(14), fontWeight: 700, zIndex: 14 }}>L</div>

        <div
          style={{
            width: wp(chartOdontoLineWidth),
            height: scalePx(1),
            left: wp(chartOdontoLineLeft),
            top: hp(chartOdontoLineY),
            position: 'absolute',
            background: '#B4B4B4',
          }}
        />
        <div
          style={{
            width: scalePx(1),
            height: hp(chartOdontoVerticalHeight),
            left: wp(chartOdontoCenterX),
            top: hp(chartOdontoVerticalTop),
            position: 'absolute',
            background: '#B4B4B4',
          }}
        />
        <div style={{ width: scalePx(1), height: hp(264), left: wp(chartSectionLeft), top: hp(804), position: 'absolute', background: '#5C5C5C' }} />
        <div style={{ width: scalePx(1), height: hp(chartContentHeight), left: wp(chartLegendDividerX), top: hp(chartContentTop), position: 'absolute', background: '#5C5C5C' }} />
        <div style={{ width: scalePx(1), height: hp(250), left: wp(rightEdge - 1), top: hp(813), position: 'absolute', background: '#5C5C5C' }} />
        <div style={{ width: wp(chartLegendDividerX - chartSectionLeft), height: hp(chartContentHeight), left: wp(chartSectionLeft), top: hp(chartContentTop), position: 'absolute', border: `${scalePx(1)} solid #4C4C4C`, pointerEvents: 'none' }} />

        {upperLeftOrders.map((order, index) => {
          const width = Number(upperSizes[order - 1].split(' ')[0]);
          const height = Number(upperSizes[order - 1].split(' ')[1]);
          return (
            <ToothSlotImage
              key={`upper-left-${order}`}
              arch="U"
              order={order}
              left={toothSlotCentersLeft[index] - width / 2}
              top={upperBaseline - height}
              width={width}
              height={height}
              flipX
            />
          );
        })}
        {upperRightOrders.map((order, index) => {
          const width = Number(upperSizes[order - 1].split(' ')[0]);
          const height = Number(upperSizes[order - 1].split(' ')[1]);
          return (
            <ToothSlotImage
              key={`upper-right-${order}`}
              arch="U"
              order={order}
              left={toothSlotCentersRight[index] - width / 2}
              top={upperBaseline - height}
              width={width}
              height={height}
            />
          );
        })}
        {lowerLeftOrders.map((order, index) => {
          const width = Number(lowerSizes[order - 1].split(' ')[0]);
          const height = Number(lowerSizes[order - 1].split(' ')[1]);
          return (
            <ToothSlotImage
              key={`lower-left-${order}`}
              arch="L"
              order={order}
              left={toothSlotCentersLeft[index] - width / 2}
              top={lowerTop}
              width={width}
              height={height}
              flipX
            />
          );
        })}
        {lowerRightOrders.map((order, index) => {
          const width = Number(lowerSizes[order - 1].split(' ')[0]);
          const height = Number(lowerSizes[order - 1].split(' ')[1]);
          return (
            <ToothSlotImage
              key={`lower-right-${order}`}
              arch="L"
              order={order}
              left={toothSlotCentersRight[index] - width / 2}
              top={lowerTop}
              width={width}
              height={height}
            />
          );
        })}

        <div style={{ width: wp(chartLegendWidth), height: hp(chartLegendHeight), left: wp(chartLegendLeft + 22), top: hp(chartLegendTop), position: 'absolute' }}>
          {legendItems.map((item) => (
            <div key={item.label}>
              <div
                style={{
                  width: relativePercent(19, chartLegendWidth),
                  height: relativePercent(19, chartLegendHeight),
                  left: 0,
                  top: relativePercent(item.top - 834, chartLegendHeight),
                  position: 'absolute',
                  background: '#808181',
                }}
              />
              <div
                style={{
                  width: relativePercent(52, chartLegendWidth),
                  height: relativePercent(19, chartLegendHeight),
                  left: relativePercent(chartLegendWidth - 52 - 12, chartLegendWidth),
                  top: relativePercent(item.top - 834, chartLegendHeight),
                  position: 'absolute',
                  background: item.color,
                }}
              />
              <div
                style={{
                  left: relativePercent(30, chartLegendWidth),
                  top: relativePercent(item.top - 834, chartLegendHeight),
                  position: 'absolute',
                  color: 'white',
                  fontSize: scalePx(13),
                  fontWeight: 700,
                }}
              >
                {item.label}
              </div>
            </div>
          ))}
        </div>

          <button
            type="button"
            onClick={() => setIsReportActive((current) => !current)}
            aria-pressed={isReportActive}
            aria-label="Open report modal"
            style={{
              width: wp(88),
              height: hp(88),
              left: wp(reportLeft),
              top: hp(reportTop),
              position: 'absolute',
              zIndex: 30,
              borderRadius: '50%',
              outline: isReportActive ? `${scalePx(3)} solid #00C0F3` : 'none',
              outlineOffset: scalePx(2),
              boxShadow: isReportActive ? `0 0 ${scalePx(18)} rgba(0, 192, 243, 0.45)` : 'none',
              cursor: 'pointer',
            }}
          >
            <img
              style={{
                width: '100%',
                height: '100%',
                display: 'block',
                filter: isReportActive ? 'drop-shadow(0 0 10px rgba(0, 192, 243, 0.7))' : 'none',
              }}
              src={isReportActive ? displayReportButtonIcons.active : displayReportButtonIcons.inactive}
              alt=""
            />
          </button>
        </div>
      </div>
    </div>
  );
}
