import cv2
import os
import numpy as np

def detect_iris(frame, eye_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    if len(eyes) == 0:
        return None
    for (ex, ey, ew, eh) in eyes:
        iris = frame[ey:ey + eh, ex:ex + ew]
        return iris
    return None

def compare_iris(iris1, iris2):
    iris1_resized = cv2.resize(iris1, (100, 100))
    iris2_resized = cv2.resize(iris2, (100, 100))
    diff = np.sum((iris1_resized - iris2_resized) ** 2)
    mse = diff / float(100 * 100)
    return mse

def register_iris(save_folder, person_id):
    cap = cv2.VideoCapture(0)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("مشکل در دسترسی به دوربین.")
            break

        iris = detect_iris(frame, eye_cascade)

        cv2.imshow('Iris Registration - Press "s" to save', frame)

        if iris is not None and cv2.waitKey(1) & 0xFF == ord('s'):
            save_path = os.path.join(save_folder, f"{person_id}_iris.jpg")
            cv2.imwrite(save_path, iris)
            print(f"عنبیه فرد {person_id} ذخیره شد.")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def login_with_iris(registered_users_folder):
    cap = cv2.VideoCapture(0)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    registered_users = [f for f in os.listdir(registered_users_folder) if f.endswith('_iris.jpg')]

    if len(registered_users) == 0:
        print("هیچ کاربری ثبت‌نام نکرده است.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("مشکل در دسترسی به دوربین.")
            break

        iris = detect_iris(frame, eye_cascade)

        if iris is not None:
            iris_gray = cv2.cvtColor(iris, cv2.COLOR_BGR2GRAY)

            for user in registered_users:
                user_iris_path = os.path.join(registered_users_folder, user)
                saved_iris = cv2.imread(user_iris_path, cv2.IMREAD_GRAYSCALE)

                if saved_iris is None:
                    continue

                mse = compare_iris(iris_gray, saved_iris)

                if mse < 1000:
                    user_id = user.split('_')[0]
                    print(f"ورود موفق برای کاربر: {user_id}")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            print("عنبیه مطابقت ندارد.")

        cv2.imshow('Iris Login - Press "q" to quit', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    save_folder = 'saved_iris_images'

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    while True:
        print("\n--- سیستم ثبت‌نام و ورود با تشخیص عنبیه ---")
        print("1. ثبت‌نام")
        print("2. ورود")
        print("3. خروج")

        choice = input("انتخاب کنید: ")

        if choice == '1':
            person_id = input("شناسه کاربر (برای ثبت‌نام): ")
            register_iris(save_folder, person_id)

        elif choice == '2':
            print("ورود با تشخیص عنبیه...")
            login_with_iris(save_folder)

        elif choice == '3':
            print("خروج از سیستم.")
            break

        else:
            print("انتخاب نامعتبر. دوباره تلاش کنید.")

if __name__ == "__main__":
    main()