
# Debug Script to Verify Response Construction Logic
import json

# MOCK DATA simulating PanoPipeline output
mock_teeth = [
    {"tooth_label": "11", "box": [100, 100, 150, 150]},
    {"tooth_label": "12", "box": [160, 100, 210, 150]},
    {"tooth_label": "21", "box": [220, 100, 270, 150]},
]

mock_caries = [
    {"label": "Caries", "assigned_tooth": "11", "box": [110, 110, 120, 120], "confidence": 0.9},
    {"label": "Periapical Lesion", "assigned_tooth": "21", "box": [230, 230, 240, 240], "confidence": 0.85},
    {"label": "Caries", "assigned_tooth": "99", "box": [0,0,10,10], "confidence": 0.5} # Unassigned checking
]

mock_bonelevel = {}

def simulate_route_logic():
    print("--- Simulating Route Logic ---")
    
    teeth_map = {t['tooth_label']: t for t in mock_teeth if t.get('tooth_label')}
    
    # Map Caries
    for c in mock_caries:
            assigned = c.get('assigned_tooth')
            lbl = c.get('label', 'caries').lower()
            
            print(f"Processing Finding: {lbl} -> assigned: {assigned}")
            
            if assigned and assigned in teeth_map:
                if 'findings' not in teeth_map[assigned]: teeth_map[assigned]['findings'] = []
                finding_type = 'periapical' if 'perio' in lbl or 'periapical' in lbl else 'caries'
                
                teeth_map[assigned]['findings'].append({'type': finding_type, 'box': c['box'], 'conf': c['confidence']})
                teeth_map[assigned][finding_type] = True
                print(f"  Mapped to {assigned} as {finding_type}")
            else:
                print(f"  Failed to map: Tooth {assigned} not found in teeth_map")

    # Serialize check
    frontend_analysis_result = {
        "teeth": teeth_map,
        "caries": mock_caries
    }
    
    print("\n--- Resulting Teeth Map ---")
    print(json.dumps(teeth_map, indent=2))
    
    return frontend_analysis_result

if __name__ == "__main__":
    simulate_route_logic()
