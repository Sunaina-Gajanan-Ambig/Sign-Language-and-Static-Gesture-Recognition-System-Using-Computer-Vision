import requests
from PIL import Image
from io import BytesIO

# URL of the image
url = "https://th.bing.com/th/id/OIP.RJVgmtIsCsI_DeIOruTHDwHaI4?w=750&h=900&rs=1&pid=ImgDetMain"

# Fetch the image using requests
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Open the image using PIL
    img = Image.open(BytesIO(response.content))
    img.show()
else:
    print("Failed to retrieve the image.")
