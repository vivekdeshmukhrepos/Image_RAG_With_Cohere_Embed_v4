class Configuration:
    Cohere_API_key = 'Cohere_API_key' # Your API key from https://dashboard.cohere.com/api-keys
    Gemini_API_key = 'Gemini_API_key' # https://aistudio.google.com/app/apikey
    # Sample images from https://www.appeconomyinsights.com/
    Images = {
        "alphabet.png": "https://substackcdn.com/image/fetch/w_600,h_400,c_fill,f_webp,q_auto:good,fl_progressive:steep,g_center/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F54f2a624-cc36-467f-b51d-473751e424cd_2744x1539.png",
        "nike.png": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa5cd33ba-ae1a-42a8-a254-d85e690d9870_2741x1541.png",
    }
    Image_Folder_Name="Resources\Images"
    MAX_PIXELS = 1568 * 1568  # Maximum allowed pixels for image resizing to control memory usage
