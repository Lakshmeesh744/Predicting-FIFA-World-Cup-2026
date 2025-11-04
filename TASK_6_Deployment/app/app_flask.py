from flask import Flask, render_template, request, jsonify
import json
import os
from enhanced_predictor import EnhancedPredictor
from data_loader import get_data_loader

# Get the correct template folder path (TASK_6_Deployment/templates)
current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, "..", "templates")

app = Flask(__name__, template_folder=template_dir)

# Load teams data from scraped sources (FIFA rankings + player database)
print(" Loading FIFA 2026 data from scraped sources...")
data_loader = get_data_loader()
teams_data = data_loader.get_48_teams_data()
print(f" Loaded {len(teams_data)} teams with real FIFA data!\n")

@app.route('/')
def home():
    return render_template('index_flask.html', teams=teams_data)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    team1_name = data.get('team1')
    team2_name = data.get('team2')
    
    # Find teams
    team1 = next((t for t in teams_data if t['name'] == team1_name), None)
    team2 = next((t for t in teams_data if t['name'] == team2_name), None)
    
    if not team1 or not team2:
        return jsonify({'error': 'Teams not found'}), 400
    
    # Use enhanced prediction model
    prediction = EnhancedPredictor.calculate_win_probability(team1, team2)
    
    # Get human-readable explanations
    explanations = EnhancedPredictor.get_factor_explanation(
        prediction['factors'],
        team1_name,
        team2_name
    )
    
    return jsonify({
        'team1': team1,
        'team2': team2,
        'team1_probability': prediction['team1_probability'],
        'team2_probability': prediction['team2_probability'],
        'winner': prediction['winner'],
        'winner_probability': prediction['winner_probability'],
        'confidence': prediction['confidence'],
        'explanations': explanations,
        'factors': prediction['factors']
    })

@app.route('/refresh-data', methods=['POST'])
def refresh_data():
    """
    Manual endpoint to refresh player data from web scraper
    """
    try:
        global teams_data, data_loader
        
        print("\n Manual data refresh requested...")
        
        # Run scraper
        success = data_loader.refresh_player_data()
        
        if success:
            # Reload teams data
            teams_data = data_loader.get_48_teams_data()
            print(f" Refreshed! Loaded {len(teams_data)} teams\n")
            
            return jsonify({
                'success': True,
                'message': 'Player data refreshed successfully!',
                'teams_count': len(teams_data)
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Refresh completed, using existing data',
                'teams_count': len(teams_data)
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/team-details/<team_name>', methods=['GET'])
def team_details(team_name):
    """
    Get detailed team information including top players (FIFA 22 onwards)
    """
    try:
        # Find team in teams_data
        team = next((t for t in teams_data if t['name'] == team_name), None)
        
        if not team:
            return jsonify({'error': 'Team not found'}), 404
        
        # Get player stats for this team (FIFA 22 onwards - year >= 2022)
        player_stats = data_loader.get_player_stats_for_team(team_name, min_year=2022)
        
        response = {
            'team': team,
            'players': player_stats
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print(" FIFA World Cup 2026 Predictor")
    print("="*60)
    print(" Server starting...")
    print(" URL: http://localhost:5000")
    print("="*60)
    print("\n Press Ctrl+C to stop the server\n")
    app.run(debug=True, port=5000, host='0.0.0.0')
