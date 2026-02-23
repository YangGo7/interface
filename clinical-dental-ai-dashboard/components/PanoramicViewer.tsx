
import React from 'react';

interface PanoramicViewerProps {
  selectedToothId: number;
  showOverlay: boolean;
  onSelectTooth: (id: number) => void;
}

const PanoramicViewer: React.FC<PanoramicViewerProps> = ({ selectedToothId, showOverlay, onSelectTooth }) => {
  return (
    <div className="flex-1 relative bg-black flex items-center justify-center overflow-hidden group/pan">
      <div 
        className="relative w-full h-full max-h-[60vh] aspect-[2/1] bg-contain bg-center bg-no-repeat transition-all duration-700"
        style={{ 
          backgroundImage: `url('https://lh3.googleusercontent.com/aida-public/AB6AXuAymd1DtkJGuCX3SeAX1BKeFY8X6sEX2WfsP74m4wzTVnecpPIutl5iJ1-9DO49pN_jUvxfErDqBonHK4olvISYHB4bK98xAsjmoG34N1mERklEKnPRq0uc2CebHb3DNIXIfbf9cgxHbsIxeQfXRm5Kh1jrnN13QJOx0eMqIdG8Qy7QkQzFGroyXSy-CfHXogB-eZUJjyoDhUVXGL_C1Rz4Bi52_igMg0349aH_RNd6VWWW1bENkEUFJWMu7F5IG1AGOrVYxpM-dtQ')`,
          filter: showOverlay ? 'none' : 'grayscale(1) contrast(1.2)'
        }}
      >
        {showOverlay && (
          <>
            {/* Box for FDI #36 (Lower Left 1st Molar) */}
            <div 
              onClick={() => onSelectTooth(36)}
              className={`absolute top-[40%] left-[25%] w-[5%] h-[15%] border-2 cursor-pointer transition-all ${
                selectedToothId === 36 ? 'border-white bg-white/10' : 'border-[#135bec]/70 hover:bg-blue-500/10'
              }`}
            >
              <div className="bg-[#135bec] text-white text-[10px] font-bold px-1 absolute -bottom-4 left-0">#36</div>
            </div>

            {/* Box for FDI #46 (Lower Right 1st Molar) */}
            <div 
              onClick={() => onSelectTooth(46)}
              className={`absolute top-[42%] right-[22%] w-[4%] h-[14%] border-2 cursor-pointer transition-all ${
                selectedToothId === 46 ? 'border-white bg-white/10 ring-4 ring-red-500/20' : 'border-red-500/70 hover:bg-red-500/10'
              }`}
            >
              <div className="bg-red-500 text-white text-[10px] font-bold px-1 absolute -bottom-4 left-0">#46</div>
            </div>

            {/* Box for FDI #11 (Upper Right Central Incisor) */}
            <div 
              onClick={() => onSelectTooth(11)}
              className={`absolute top-[38%] left-[50%] -translate-x-1/2 w-[4%] h-[12%] border-2 cursor-pointer transition-all ${
                selectedToothId === 11 ? 'border-white bg-white/10' : 'border-yellow-500/70 hover:bg-yellow-500/10'
              }`}
            >
              <div className="bg-yellow-500 text-black text-[10px] font-bold px-1 absolute -bottom-4 left-0">#11</div>
            </div>
          </>
        )}
      </div>

      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-3 px-4 py-2 bg-black/60 backdrop-blur-md rounded-full border border-white/10 text-white text-[11px] opacity-0 group-hover/pan:opacity-100 transition-opacity">
        <span>FDI Standard View</span>
        <span className="w-px h-3 bg-white/20"></span>
        <span>W: 2400 / L: 1200</span>
      </div>
    </div>
  );
};

export default PanoramicViewer;
