import boto3
import pandas as pd
import warnings
from tabulate import tabulate
from termcolor import colored

warnings.simplefilter(action="ignore", category=FutureWarning)

"""
# This code is:
# 1. Detecting faces in an image
# 2. Getting the emotion of the face
# 3. Getting the age range of the face
# 4. Getting their attributes
# 5. Creating a dictionary of the face details
# 6. The next function is converting the dictionary to a dataframe
# 7. Displaying the dataframe in a table with columns 'Feauture', 'Value', 'Confidence'
"""


def readImage(file):
    """
    Reads the image file and returns the bytes
    :param file: The image file
    :return: The bytes of the image
    """
    with open(file, "rb") as image:
        return bytearray(image.read())


def analyze_sentiment_in_face(image):
    """
    Detects the faces in the image and returns the face details
    :param image: The image file
    :return: The face details of the image
    """

    client = boto3.client("rekognition")
    return client.detect_faces(Image={"Bytes": readImage(image)}, Attributes=["ALL"])


def get_emotion(fd):
    """
    Returns the emotion of the face
    :param fd: The face details of the face

    :return: The emotion of the face
    """

    return fd["FaceDetails"][0]["Emotions"][0]["Type"]


def get_age_range(fd):
    """
    Returns the age range of the face
    :param fd: The face details of the face

    :return: The age range of the face
    """

    return next(
        (emotion["Type"] for emotion in fd["Emotions"] if emotion["Confidence"] > 80),
        emotions,
    )


def get_age_range(fd):
    """
    Returns the age range of the face
    :param fd: The face details of the face

    :return: The age range of the face
    """
    return fd["AgeRange"]


def get_smile(fd):
    """
    Returns the smile of the face
    """
    return fd["Smile"]


def get_gender(fd):
    """
    Returns the gender of the face
    """
    return fd["Gender"]


def get_beard(fd):
    """
    Returns the beard of the face
    """
    return fd["Beard"]


def get_mustache(fd):
    """
    Returns the mustache of the face
    """
    return fd["Mustache"]


def get_eyeglasses(fd):
    """
    Returns the eyeglasses of the face
    """
    return fd["Eyeglasses"]


def get_sunglasses(fd):
    """
    Returns the sunglasses of the face
    """
    return fd["Sunglasses"]


def get_eyes_open(fd):
    """
    Returns the eyes open value from the face details
    :param fd: The face details

    :return: The eyes open value
    """
    return fd["EyesOpen"]


def get_mouth_open(fd):
    """
    Returns the mouth open value from the face details
    :param fd: The face details
    :return: The mouth open value
    """
    return fd["MouthOpen"]


def get_emotions(fd):
    """
    Returns the emotions from the face details
    :param fd: The face details
    :return: The emotions
    """
    return fd["Emotions"]


def get_landmarks(fd):
    """
    Returns the landmarks from the face details
    :param fd: The face details
    :return: The landmarks
    """
    return pd.DataFrame(fd["Landmarks"])


def get_pose(fd):
    """
    Returns the pose from the face details
    :param fd:
    :return:
    """
    return pd.DataFrame(fd["Pose"])


def get_quality(fd):
    """
    Returns the quality from the face details
    :param fd:
    :return:
    """
    return pd.DataFrame(fd["Quality"])


def get_confidence(fd):
    """
    Returns the confidence from the face details
    :param fd:
    :return:
    """
    return fd["Confidence"]


def get_fd(fd):
    """
    Returns the face details from the face details
    :param fd:
    :return:
    """
    return {
        "age_range": get_age_range(fd),
        "smile": get_smile(fd),
        "gender": get_gender(fd),
        "beard": get_beard(fd),
        "mustache": get_mustache(fd),
        "eyeglasses": get_eyeglasses(fd),
        "sunglasses": get_sunglasses(fd),
        "eyes_open": get_eyes_open(fd),
        "mouth_open": get_mouth_open(fd),
        "emotions": get_emotions(fd),
        "confidence": get_confidence(fd),
    }


def get_landmark_details(landmark_details):
    """
    Returns the landmark details
    :param landmark_details: The landmark details

    :return: The landmark details
    """
    return {
        "type": landmark_details["Type"],
        "x": landmark_details["X"],
        "y": landmark_details["Y"],
    }


def get_pose_details(pose_details):
    """
    Returns the pose details
    :param pose_details: The pose details

    :return: The pose details
    """
    return {
        "pitch": pose_details["Pitch"],
        "roll": pose_details["Roll"],
        "yaw": pose_details["Yaw"],
    }


def get_quality_details(quality_details):
    """
    Returns the quality details
    :param quality_details: The quality details

    :return: The quality details
    """
    return {
        "brightness": quality_details["Brightness"],
        "sharpness": quality_details["Sharpness"],
    }


def convert_confidence_to_percentage(df):
    """
    Converts the confidence to percentage
    :param df: The dataframe
    :return: The dataframe with the confidence converted to percentage
    """
    return df["Confidence"].apply(lambda x: f"{round(x)}%")


def convert_fd_to_dataframe(fd):
    """
    Converts the face details to a dataframe
    :param fd: The face details
    :return: The dataframe
    """
    headers = ["Feature", "Value", "Confidence"]

    fd = get_fd(fd)
    df = pd.DataFrame(columns=headers)

    for key, value in fd.items():
        if key == "emotions":
            for emotion in value:
                if emotion["Confidence"] > 80:
                    df = df.append(
                        {
                            "Feature": key,
                            "Value": emotion["Type"],
                            "Confidence": emotion["Confidence"],
                        },
                        ignore_index=True,
                    )
        elif key == "age_range":
            df = df.append(
                {
                    "Feature": key,
                    "Value": f'{value["Low"]} - {value["High"]}',
                    "Confidence": fd["confidence"],
                },
                ignore_index=True,
            )
        elif key in [
            "smile",
            "gender",
            "beard",
            "mustache",
            "eyeglasses",
            "sunglasses",
            "eyes_open",
            "mouth_open",
        ]:
            df = df.append(
                {
                    "Feature": key,
                    "Value": value["Value"],
                    "Confidence": value["Confidence"],
                },
                ignore_index=True,
            )

        elif key == "confidence":
            df = df.append(
                {
                    "Feature": "confidence",
                    "Value": f"{round(value, 2)}%",
                    "Confidence": value,
                },
                ignore_index=True,
            )
        else:
            df = df.append(
                {"Feature": key, "Value": value, "Confidence": fd["confidence"]},
                ignore_index=True,
            )

    df.style.set_properties(**{"text-align": "left"})

    print()

    print()
    print()
    print("Face Details")
    print(tabulate(df, headers="keys", tablefmt="psql"))
    df.style.set_properties(**{"text-align": "left"})
    print("Face Details")

    pd.set_option("display.colheader_justify", "left")

    df["Confidence"] = convert_confidence_to_percentage(df)

    return df


def convert_landmarks_to_df(fd):
    """
    Converts the landmarks to a dataframe
    :param fd: The face details
    :return: The dataframe
    """
    headers = ["Feature", "X", "Y"]
    df = pd.DataFrame(columns=headers)
    for landmark in fd["Landmarks"]:
        landmark_details = get_landmark_details(landmark)
        df = df.append(
            {
                "Feature": landmark_details["type"],
                "X": landmark_details["x"],
                "Y": landmark_details["y"],
            },
            ignore_index=True,
        )
    df.style.set_properties(**{"text-align": "left"})
    print()
    print("Landmarks")
    print(tabulate(df, headers="keys", tablefmt="psql"))
    return df


def convert_pose_to_df(fd):
    """
    Converts the pose to a dataframe
    :param fd: The face details
    :return: The dataframe
    """
    headers = ["Feature", "Value"]
    df = pd.DataFrame(columns=headers)
    pose_details = get_pose_details(fd["Pose"])
    for key, value in pose_details.items():
        df = df.append({"Feature": key, "Value": value}, ignore_index=True)
    df.style.set_properties(**{"text-align": "left"})
    print()
    print("Pose")
    print(tabulate(df, headers="keys", tablefmt="psql"))
    return df


def convert_quality_to_df(fd):
    """
    Converts the quality to a dataframe
    :param fd: The face details
    :return: The dataframe
    """
    headers = ["Feature", "Value"]
    df = pd.DataFrame(columns=headers)
    quality_details = get_quality_details(fd["Quality"])
    for key, value in quality_details.items():
        df = df.append({"Feature": key, "Value": value}, ignore_index=True)
    df.style.set_properties(**{"text-align": "left"})
    print()
    print("Quality")
    print(tabulate(df, headers="keys", tablefmt="psql"))
    return df


if __name__ == "__main__":
    image = "maxwell.png"
    face = analyze_sentiment_in_face(image)["FaceDetails"][0]
    convert_landmarks_to_df(face)
    convert_pose_to_df(face)
    convert_quality_to_df(face)
    convert_fd_to_dataframe(face)
