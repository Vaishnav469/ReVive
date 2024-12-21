import React, { useEffect, useState } from "react";
import { useRouter } from "expo-router";
import { View, Text, TextInput, Button, StyleSheet, ActivityIndicator } from "react-native";
import { useOnboarding } from "./_layout";
import { useLocalSearchParams } from 'expo-router';
import { useTheme } from "@react-navigation/native";

const QuestionScreen = () => {
  const { key } = useLocalSearchParams(); // Still get the key from params if needed
  const { handleResponse, questions } = useOnboarding();
  const [response, setResponse] = useState("");
  const { colors } = useTheme();
  const questionObj = questions.find((q) => q.key === key);
  const question = questionObj ? questionObj.question : "No question found";


  useEffect(() => {
    setResponse(""); // Clear the response for the new question
  }, [key]);

  const handleNext = () => {
    if (!response.trim()) {
      alert("Please provide an answer before proceeding.");
      return;
    }


    handleResponse(key, response); // Save response and navigate
  };

  return (
        <View style={[styles.container,  { backgroundColor: colors.background }]}>
          <Text style={[styles.text, { color: colors.text }]}>{question}</Text>
          <TextInput
            style={[styles.input,
              {
                color: colors.text,
                borderColor: colors.border,
                backgroundColor: colors.card, // Ensures contrast for input
              },
            ]}
            value={response}
            onChangeText={setResponse}
            placeholder="Type your answer here"
            placeholderTextColor={colors.text || "#aaa"}
          />
          <Button
            title="Next"
            onPress={() => handleNext()}
            color={colors.primary}
          />
        </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    padding: 20,
  },
  input: {
    height: 40,
    borderWidth: 1,
    borderColor: "#ddd",
    borderRadius: 5,
    marginBottom: 15,
    paddingHorizontal: 10,
  },
  text: {
    fontSize: 16,
    marginBottom: 15,

  }
});

export default QuestionScreen;
