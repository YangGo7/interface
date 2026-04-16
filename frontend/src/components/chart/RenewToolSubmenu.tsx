type RenewToolSubmenuItem = {
  key: string;
  label: string;
  active?: boolean;
  onClick: () => void;
};

type RenewToolSubmenuProps = {
  visible: boolean;
  left: string;
  top: string;
  width?: string;
  items: RenewToolSubmenuItem[];
  onClose?: () => void;
};

export function RenewToolSubmenu({
  visible,
  left,
  top,
  width = '116px',
  items,
  onClose,
}: RenewToolSubmenuProps) {
  if (!visible) return null;

  return (
    <div
      onPointerDown={(event) => {
        event.stopPropagation();
      }}
      onClick={(event) => {
        event.stopPropagation();
      }}
      style={{
        position: 'absolute',
        left,
        top,
        width,
        background: '#2D2D2D',
        border: '1px solid #4C4C4C',
        boxShadow: '2px 2px 0 rgba(0, 0, 0, 0.35)',
        zIndex: 20,
        overflow: 'hidden',
      }}
    >
      {items.map((item) => (
        <button
          key={item.key}
          type="button"
          onClick={() => {
            item.onClick();
            onClose?.();
          }}
          style={{
            width: '100%',
            minHeight: '34px',
            padding: '0 10px',
            border: 'none',
            borderBottom: '1px solid #3A3A3A',
            background: item.active ? '#414141' : '#2D2D2D',
            color: '#FFFFFF',
            fontSize: '11px',
            fontWeight: item.active ? 700 : 500,
            textAlign: 'center',
            cursor: 'pointer',
          }}
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}
