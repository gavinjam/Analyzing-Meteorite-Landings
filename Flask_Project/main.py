from flask import Flask, render_template, request, jsonify, send_file, flash
import requests
from PyAnalysis import meteorite_landings as ml
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.externals import joblib

app = Flask(__name__)
app.secret_key = " super secret "

pdata = pd.read_csv('C:/Users/gavin/GAVIN_STASH/Personal_Project_Experiment/Flask_Project/PyAnalysis/Meteorite_Landings.csv', na_values=[])
pdata = ml.timestamp_to_year(pdata)
gdata = ml.pd_to_gpd(pdata)
countries = gpd.read_file('C:/Users/gavin/GAVIN_STASH/Personal_Project_Experiment/Flask_Project/PyAnalysis/ne_110m_admin_0_countries.shp')
merged = ml.merged_dataset(gdata, countries)

jsonPopRareClass = {"popular": ml.popular_class(pdata), "rare": ml.rarest_class(pdata)}

@app.route("/")
def test():
    return ml.test_hello()

@app.route("/index")
def index():
    flash("This is a flashed message.")
    return render_template("index.html")

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/nums")
def nums():    
    response = requests.get("http://127.0.0.1:5000/api/v1/poprareclass")
    json_response = response.json()
    popular_class_type = json_response['popular']
    rare_class_type = json_response['rare']

    popular_class_loc_value = ml.popular_place_of_popular_class(merged, popular_class_type)
    average_mass_value = ml.avg_mass(pdata)
    range_mass_value = ml.range_mass(pdata)
    big_mass_value = ml.biggest_mass(pdata)
    small_mass_value = ml.smallest_mass(pdata)

    return render_template("numericalanalysis.html", popular_class=popular_class_type,
                            popular_class_loc=popular_class_loc_value,
                            rarest_class=rare_class_type, average_mass=average_mass_value,
                            range_mass=range_mass_value, big_mass=big_mass_value,
                            small_mass=small_mass_value)

@app.route('/popclassmap')
def pop_class_map():
    response = requests.get("http://127.0.0.1:5000/api/v1/poprareclass")
    json_response = response.json()
    popular_class_type = json_response['popular']

    countries = gpd.read_file('C:/Users/gavin/GAVIN_STASH/Personal_Project_Experiment/Flask_Project/PyAnalysis/ne_110m_admin_0_countries.shp')

    return ml.plot_popular_class(gdata, countries, popular_class_type)

@app.route('/rareclassmap')
def rare_class_map():
    response = requests.get("http://127.0.0.1:5000/api/v1/poprareclass")
    json_response = response.json()
    rare_class_type = json_response['rare']

    countries = gpd.read_file('C:/Users/gavin/GAVIN_STASH/Personal_Project_Experiment/Flask_Project/PyAnalysis/ne_110m_admin_0_countries.shp')

    return ml.plot_rarest_class(gdata, countries, rare_class_type)

@app.route('/massovertimemap')
def mass_overtime_map():
    return ml.plot_mass_overtime(pdata)

@app.route('/allmassesmap')
def all_masses_map():
    countries = gpd.read_file('C:/Users/gavin/GAVIN_STASH/Personal_Project_Experiment/Flask_Project/PyAnalysis/ne_110m_admin_0_countries.shp')
    return ml.plot_all_mass_map(gdata, countries)

@app.route('/biggermassesmap')
def bigger_masses_map():
    countries = gpd.read_file('C:/Users/gavin/GAVIN_STASH/Personal_Project_Experiment/Flask_Project/PyAnalysis/ne_110m_admin_0_countries.shp')
    return ml.plot_bigger_mass_map(gdata, countries)

@app.route("/graphs")
def graphs():
    return render_template("graphs.html")

@app.route("/api/v1/poprareclass", methods=['GET'])
def Popular_Rare_Class_Types_json():
    return jsonify(jsonPopRareClass)

@app.route("/predict", methods=('GET', 'POST'))
def predict():
    accuracy_score = int(ml.predict_place_of_meteorite_impact(merged) * 100 )
    mass = None
    year = None

    if request.method == 'POST':
        mass = request.form['mass']
        year = request.form['year']

    # exit a flash message or fade-out
    if (not mass) or (not year):
        flash('Please Enter a Mass and Year Value', 'info')
        return render_template("predictionform.html", accuracy=accuracy_score)
    
    # Error codes
    # Error Handling Case: If something else is wrong, flash the Sys.exit(1)
    #                      error and after a certain amount of counts of
    #                      returning the same html template to reload, quit or something.
    else:
        try:
            mass = int(mass)
            year = int(year)
        except ValueError:
            flash('One or more of the fields are not valid integers', 'error')
            return render_template("predictionform.html", accuracy=accuracy_score)

        flash('Predicted Landing Location: ' + ml.predict(mass, year), 'message')
        flash('Please Enter a new Mass and Year Values', 'info')

    return render_template("predictionform.html", accuracy=accuracy_score)

# Does it make sense of the user to enter whatever? How to validate it?
@app.route("/addtodata")
def add_to_data():
    return render_template("addtodata.html")

# HTML fetch(get request passing username and id)
# /test/api/{username}
@app.route("/api/v1/<username>")
def test_params(username):
    return username

@app.route("/api/v2/<username>")
def test_query(username):
    id = request.args['id']
    return username + ' ' + str(id)

@app.errorhandler(404)
def page_not_found(e):
    return render_template("pagenotfound.html"), 404
    
if __name__ == "__main__":
    app.run(debug=True)
