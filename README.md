# 🌱 ReVive: Transform Household Items into Creative Masterpieces!  

ReVive is an innovative upcycling app that encourages sustainability by providing creative upcycling ideas for household items. Whether you're looking to repurpose an old jar or transform a cardboard box into something extraordinary, ReVive has you covered! 

---

## 🌟 Inspiration

The inspiration behind ReVive was to promote a sustainable lifestyle by encouraging individuals to breathe new life into old household items. With machine learning and AI, ReVive bridges the gap between environmental consciousness and creativity.

---

## 🚀 How It Works

1. **Upload or Capture an Image**: Users can upload a picture of a household item or use their camera to take a photo.
2. **Prediction**: Our fine-tuned ResNet50 model predicts the item in the image (trained on 25 different labels with 90% accuracy).
3. **AI Suggestions:** The app automatically uses GPT-4 to generate personalized upcycling ideas based on the prediction and user preferences (location, time, and budget that is asked when signed up).
4. **Manual Input:** If the user thinks the prediction is wrong, they can manually enter the item they want to upcycle to receive updated suggestions.
5. **YouTube Integration**: The app fetches related YouTube videos using the YouTube API to provide visual inspiration for the upcycling process.
6. **Frontend Display**: All suggestions and videos are presented in an easy-to-navigate interface for users to explore.

---

## 🛠️ How We Built It

- **Machine Learning**: Fine-tuned a ResNet50 model in PyTorch on a dataset of 25 household item labels, achieving 90% accuracy.
- **Backend**: Flask API to connect the ML model and handle user requests.
- **Frontend**: Built with React Native and styled with CSS for a seamless user experience.
- **Personalization**: OpenAI's GPT-4 for generating personalized upcycling ideas based on user inputs (location, time, budget).
- **YouTube API**: Integrated to fetch related upcycling video tutorials.
- **Database**: Firebase for user authentication and data storage.

---

## 🏆 Accomplishments

- Successfully fine-tuned ResNet50 to achieve 90% accuracy on a custom dataset.
- Developed a fully functional app that combines computer vision, natural language processing, and third-party APIs.
- Created a personalized user experience by considering location, time, and budget in upcycling suggestions.
- Seamlessly integrated YouTube videos into the app for additional guidance.

---

## 🔮 What's Next

- **Enhanced Dataset**: Expand the dataset to include more household items for better predictions.
- **Community Features**: Allow users to share their upcycling projects and inspire others.
- **Real-Time Suggestions**: Implement a feature to provide real-time step-by-step guidance for upcycling projects.
- **Gamification**: Add rewards and challenges to motivate users to upcycle more frequently.
- **Localization:** Add support for multiple languages to reach a global audience.
- **AR Integration:** Help users visualize their upcycled ideas before starting!

---

## 💻 Tech Stack

- **Backend**: Flask, PyTorch, OpenAI GPT-4
- **Frontend**: React Native, CSS
- **Database**: Firebase
- **APIs**: YouTube API, OpenAI API
- **Other Tools**: Matplotlib, Seaborn for data visualization during model training

---
## 🎉 Try ReVive!  

ReVive makes sustainability fun and creative. Whether it’s an old jar, a broken chair, or leftover fabric, there’s always a way to give it a new life. Let’s upcycle and make the world a little greener, together! 🌿✨  


