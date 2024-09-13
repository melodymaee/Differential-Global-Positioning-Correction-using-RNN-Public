import phonenumbers
import opencage
import folium
from phonenumbers import geocoder
number = "+639171628305"

pepnumber = phonenumbers.parse(number)

location = geocoder.description_for_number(pepnumber, "en")
print(location)

from phonenumbers import carrier

service_prov = phonenumbers.parse(number)
print(carrier.name_for_number(service_prov, "en"))

from opencage.geocoder import OpenCageGeocode

key = '08fe29c574464304802653777870d66c'

geocoder = OpenCageGeocode(key)
query = str(location)
results = geocoder.geocode(query)
#print(results)

lat = 8.2422474 #results[0]['geometry']['lat']
lng = 144.2440479 #results[0]['geometry']['lng']

print(lat,lng)

myMap = folium.Map(location=[lat,lng], zoom_start=9)
folium.Marker([lat,lng], popup=location).add_to(myMap)

myMap.save("mylocation.html")