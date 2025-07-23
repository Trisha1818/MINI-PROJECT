# import streamlit as st
# import sqlite3
# import cv2
# import numpy as np
# import os
# import matplotlib.pyplot as plt

# # Database setup
# conn = sqlite3.connect('users.db')
# cursor = conn.cursor()

# # Create table for users if not exists
# cursor.execute('''CREATE TABLE IF NOT EXISTS users
#                   (username TEXT, password TEXT, phone TEXT, email TEXT, gender TEXT, address TEXT)''')

# cursor.execute('''CREATE TABLE IF NOT EXISTS history
#                   (username TEXT, image_name TEXT, result_path TEXT, rgb_ratio TEXT)''')

# # Function to register user
# def register_user(username, password, phone, email, gender, address):
#     cursor.execute("INSERT INTO users (username, password, phone, email, gender, address) VALUES (?, ?, ?, ?, ?, ?)",
#                    (username, password, phone, email, gender, address))
#     conn.commit()

# # Function to verify login
# def verify_login(username, password):
#     cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
#     return cursor.fetchone()

# # Function to store history of images
# def store_image_history(username, image_name, result_path, rgb_ratio):
#     cursor.execute("INSERT INTO history (username, image_name, result_path, rgb_ratio) VALUES (?, ?, ?, ?)",
#                    (username, image_name, result_path, rgb_ratio))
#     conn.commit()

# # Streamlit UI setup
# st.set_page_config(page_title="Shadow Removal App", layout="centered")
# st.title("Welcome to the Shadow Removal App")

# # Sidebar for login and register
# menu = ["Login", "Register"]
# choice = st.sidebar.selectbox("Select a page", menu)

# if choice == "Login":
#     username = st.text_input("Username")
#     password = st.text_input("Password", type='password')

#     if st.button("Login"):
#         user = verify_login(username, password)
#         if user:
#             st.session_state.logged_in = True
#             st.session_state.username = username
#             st.success("Logged in successfully!")
#         else:
#             st.error("Invalid credentials. Please try again.")

# elif choice == "Register":
#     username = st.text_input("Username")
#     password = st.text_input("Password", type='password')
#     phone = st.text_input("Phone")
#     email = st.text_input("Email")
#     gender = st.selectbox("Gender", ["Male", "Female", "Other"])
#     address = st.text_area("Address")

#     if st.button("Register"):
#         register_user(username, password, phone, email, gender, address)
#         st.success("You have successfully registered!")

# # User Home Page
# if 'logged_in' in st.session_state and st.session_state.logged_in:
#     st.title(f"Welcome, {st.session_state.username}")

#     col1, col2 = st.columns(2)

#     with col1:
#         if st.button("Upload Image to Remove Shadow"):
#             uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

#             if uploaded_file is not None:
#                 # Save uploaded image
#                 image_path = os.path.join("uploads", uploaded_file.name)
#                 with open(image_path, "wb") as f:
#                     f.write(uploaded_file.getbuffer())

#                 # Apply shadow removal logic here (your image processing code)
#                 shadow_removed_image = remove_shadow(image_path)  # This should be a function to remove shadow

#                 st.image(shadow_removed_image, caption="Shadow Removed Image", use_column_width=True)

#                 # Save result and RGB ratios
#                 result_path = "results/" + uploaded_file.name
#                 rgb_ratio = calculate_rgb_ratio(shadow_removed_image)  # You should define this function
#                 store_image_history(st.session_state.username, uploaded_file.name, result_path, rgb_ratio)

#         with col2:
#             st.write("## View Image History")
#             cursor.execute("SELECT image_name, result_path FROM history WHERE username = ?", (st.session_state.username,))
#             history = cursor.fetchall()

#             if len(history) > 0:
#                 for record in history:
#                     st.write(f"Image: {record[0]} | Result: [View Image]({record[1]})")
#             else:
#                 st.write("No image history found.")


import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import closed_form_matting

# Function to compute the average value
def computeAverage(arr):
    arr[arr == 0] = np.nan
    arr_mean = np.nanmean(arr, axis=1)
    arr_mean = np.nanmean(arr_mean, axis=0)
    return arr_mean

# Function to get intensity ratio
def getIntensityRatio(imgPatch, maskPatch, softMaskPatch):
    s = imgPatch
    m = maskPatch
    shd = s * m
    non_shd = s * np.logical_not(m)

    shd_mean = computeAverage(shd)
    non_shd_mean = computeAverage(non_shd)

    shd_K = softMaskPatch * m
    non_shd_K = softMaskPatch * np.logical_not(m)

    shd_mean_K = computeAverage(shd_K)
    non_shd_mean_K = computeAverage(non_shd_K)

    r = (non_shd_mean - shd_mean) / ((shd_mean * non_shd_mean_K) - (non_shd_mean * shd_mean_K))
    
    return r

# Function to get mask ratio
def getMaskRatio(m):
    totalCount = m.shape[0] * m.shape[1]
    mOnes = np.count_nonzero(m[:, :, 0]) 
    mZeros = totalCount - mOnes
    patchRatio = mOnes / (mOnes + mZeros)
    return patchRatio

# Function to get patches
def getPatch(soft_mask, hard_mask, shadow_img, x, y, patch_size, width, height):
    xE = x + patch_size
    yE = y + patch_size
    
    if xE >= height:
        xE = xE - (xE - height + 1)
    if yE >= width:
        yE = yE - (yE - width + 1)

    soft_m = soft_mask[x:xE, y:yE, :].copy()
    m = hard_mask[x:xE, y:yE, :].copy()
    s = shadow_img[x:xE, y:yE, :].copy()

    maskRatio = getMaskRatio(m)
    if maskRatio > 0.49 and maskRatio < 0.51:
        return s, m, soft_m, True
    else:
        return None, None, None, False

# Function to update bins
def updateBins(bins, binIndx, r):
    for i in range(3):
        currentR = round(r[i], 1)
        if 0 <= int(currentR * 10) and int(currentR * 10) < len(binIndx):
            bins[int(currentR * 10), i] = bins[int(currentR * 10), i] + 1
    return bins

# Function to get the final ratio
def getFinalRatio(bins, binIndx):
    r = np.zeros(3).astype('float64')
    indx = np.where(bins == np.max(bins, axis=0))

    prevY = 0
    prevYCount = 0

    for x, y in zip(indx[0], indx[1]):
        r[y] = r[y] + binIndx[x].astype('float64')
        r = np.nan_to_num(r)

        if y != prevY:
            r[prevY] = r[prevY] / prevYCount
            prevYCount = 0

        prevY = y
        prevYCount = prevYCount + 1

    r[prevY] = r[prevY] / prevYCount
    return r

# Function to fix patch shadow
def fixPatchShadow(imgPatch, maskPatch, softMaskPatch, ratio):
    mapper = (ratio + 1) / (ratio * softMaskPatch + 1)
    fixed = imgPatch * mapper
    fg = fixed * maskPatch
    bg = fixed * np.logical_not(maskPatch)
    final = fg + bg
    return fixed

# Function to remove shadow from the image
def shadowRemover(shadow_img, soft_mask, hard_mask, patch_size=12, offset=1):
    height = shadow_img.shape[0]
    width = shadow_img.shape[1]
    binIndx = np.arange(0, 100, 0.1)
    bins = np.zeros((len(binIndx), 3))

    for x in np.arange(0, height, offset)[0:-1]:
        for y in np.arange(0, width, offset)[0:-1]:
            s, m, soft_m, check = getPatch(soft_mask, hard_mask, shadow_img, x, y, patch_size, width, height)
            if check:
                r = getIntensityRatio(s, m, soft_m)
                bins = updateBins(bins, binIndx, r)

    r = getFinalRatio(bins, binIndx)
    shadow_free_image = fixPatchShadow(shadow_img.copy(), hard_mask.copy(), soft_mask.copy(), r)
    
    return shadow_free_image, bins, r

# Streamlit App
st.title('Shadow Removal from Images')

# File uploader to upload the image
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_image:
    # Store the uploaded image in session_state
    st.session_state.image = uploaded_image
    # Read the image
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('double') / 255.0

    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process image after upload
    shadow_img = image  # Assuming this is the shadow image
    hard_mask = cv2.cvtColor(cv2.imread('Samples/HardMasks/road_shadow.png'), cv2.COLOR_BGR2RGB).astype('double') / 255.0
    scribbles = cv2.imread('Samples/Scribbles/road_shadow.png', cv2.IMREAD_COLOR) / 255.0
    soft_mask = closed_form_matting.closed_form_matting_with_scribbles(shadow_img, scribbles)

    # Save the soft mask
    cv2.imwrite('Samples/SoftMasks/road_shadow.png', soft_mask * 255.0)

    # Display the soft mask
    soft_mask = cv2.cvtColor(cv2.imread('Samples/SoftMasks/road_shadow.png'), cv2.COLOR_BGR2RGB).astype('double') / 255.0
    soft_mask = 1 - soft_mask
    st.image(soft_mask, caption="Soft Mask", use_column_width=True)

    # Remove shadow
    shadow_free_image, bins, r = shadowRemover(shadow_img, soft_mask, hard_mask, patch_size=50, offset=1)

    # Display results
    st.image(shadow_free_image, caption="Shadow Removed Image", use_column_width=True)

    # Display the RGB ratio space plot
    st.subheader("RGB Ratio Space")
    fig, ax = plt.subplots()
    ax.set_title('RGB Ratio Space')
    ax.set_xlabel("Bins")
    ax.set_ylabel("# of patches")
    ax.plot(bins)
    ax.set_xlim([0, 100])
    st.pyplot(fig)

    # Print final RGB ratios
    st.write('Final RGB Ratios:', r)
