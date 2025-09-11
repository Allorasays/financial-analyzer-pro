package com.financialanalyzer.mobile;

import android.content.Context;
import com.facebook.react.ReactInstanceManager;

/**
 * Class responsible of loading Flipper inside your React Native application. This is the release
 * flavor of {@link ReactNativeFlipper}.
 */
public class ReactNativeFlipper {
  public static void initializeFlipper(Context context, ReactInstanceManager reactInstanceManager) {
    // Do nothing as we don't want to initialize Flipper on Release.
  }
}