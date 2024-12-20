import { Text, View, Button } from "react-native";
import { FIREBASE_AUTH } from "../firebaseconfig";



const Profile = () => {
  return (
    <View
      style={{
        flex: 1,
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <Button onPress={() => FIREBASE_AUTH.signOut()} title="logout" />
      <Text>Edit app/Profile.tsx to edit this screen.</Text>
    </View>
  );
}

export default Profile;
