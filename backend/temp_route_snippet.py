
@api_v2.route('/session/<session_id>', methods=['GET'])
def get_session_status(session_id):
    if session_id not in SESSIONS:
        # If not found but we are in a permissive mode, maybe return a "not found" status instead of 404?
        # But logically it should exist if they started upload.
        return jsonify({'status': 'not_found', 'error': 'Session not found'}), 404
        
    session_data = SESSIONS[session_id]
    
    # If completed, include the full result
    response = {
        'status': session_data.get('status', 'unknown'),
        'result': session_data.get('result')
    }
    
    # If failed, include error
    if 'error' in session_data:
        response['error'] = session_data['error']
        
    return jsonify(response)
