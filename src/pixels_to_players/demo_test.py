#!/usr/bin/env python3
"""
PixelsToPlayers Demo Test - Deliverable 2 Prototype

This comprehensive test file demonstrates all the components built so far:
1. Webcam calibration for gaze tracking
2. Gaze tracking using MediaPipe face mesh
3. Screen recording functionality
4. Integration of all components working together

This serves as a prototype version of app.py to show functionality for the demo.
"""

import time
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

# Import our components
from webcam import WebcamClient, WebcamConfig, processors as P
from screen_recording import ScreenRecorder, ScreenRecordingConfig
from webcam.calibration import run_calibration, CALIB_POINTS
from webcam.tracking import run_eye_tracking
from file_operations import Logger


class PixelsToPlayersDemo:
    """
    Main demo class that integrates all components for the prototype demonstration.
    """
    
    def __init__(self):
        self.calibration_data = None
        self.recordings_dir = Path("demo_recordings")
        self.recordings_dir.mkdir(exist_ok=True)
        self.calibration_file = self.recordings_dir / "demo_calibration_data.json"
        
        # Demo configuration
        self.demo_config = {
            "webcam": WebcamConfig(
                width=640,
                height=480,
                fps=30,
                show_preview=True
            ),
            "screen_recording": ScreenRecordingConfig(
                fps=15,
                show_preview=True,
                enable_key_stop=True,
                save_dir=self.recordings_dir
            )
        }
        
        print("PixelsToPlayers Demo Initialized")
        print("=" * 50)
    
    def load_calibration_data(self) -> bool:
        """Load existing calibration data if available."""
        if self.calibration_file.exists():
            try:
                with open(self.calibration_file, 'r') as f:
                    self.calibration_data = json.load(f)
                print(f"Loaded calibration data from {self.calibration_file}")
                return True
            except Exception as e:
                print(f"Failed to load calibration data: {e}")
                return False
        return False
    
    def save_calibration_data(self, data: list) -> None:
        """Save calibration data to file."""
        try:
            with open(self.calibration_file, 'w') as f:
                json.dump(data, f, indent=2)
            self.calibration_data = data
            print(f"Calibration data saved to {self.calibration_file}")
        except Exception as e:
            print(f"Failed to save calibration data: {e}")
    
    def test_webcam_basic(self) -> bool:
        """Test basic webcam functionality with face mesh detection."""
        print("\nTesting Webcam with Face Mesh Detection")
        print("-" * 40)
        
        try:
            with WebcamClient(self.demo_config["webcam"]) as cam:
                print("Webcam opened successfully")
                print("Face mesh overlay active - press 'q' to quit")
                
                # Show live camera feed with face mesh
                while True:
                    frame = cam.snapshot(processors=[P.flip_horizontal, P.draw_facemesh])
                    cv2.imshow("Face Mesh Test", frame)
                    
                    # Check for quit key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                cv2.destroyAllWindows()
                print("Face mesh test completed")
                return True
                
        except Exception as e:
            print(f"Webcam test failed: {e}")
            return False
    
    def test_gaze_tracking(self) -> bool:
        """Test gaze tracking functionality."""
        print("\nTesting Gaze Tracking")
        print("-" * 40)
        
        try:
            with WebcamClient(self.demo_config["webcam"]) as cam:
                print("Starting gaze tracking test...")
                print("Look around and watch the iris center detection")
                print("Press 'q' to quit")
                
                start_time = time.time()
                while time.time() - start_time < 10:  # 10 second demo
                    frame = cam.snapshot(processors=[P.flip_horizontal])
                    
                    # Get iris center
                    iris_center = P.get_iris_center(frame)
                    
                    if iris_center:
                        # Draw iris center
                        cv2.circle(frame, (int(iris_center[0]), int(iris_center[1])), 5, (0, 255, 0), -1)
                        cv2.putText(frame, f"Iris: ({iris_center[0]:.1f}, {iris_center[1]:.1f})", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "No iris detected", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    cv2.imshow("Gaze Tracking Test", frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                cv2.destroyAllWindows()
                print("Gaze tracking test completed")
                return True
                
        except Exception as e:
            print(f"Gaze tracking test failed: {e}")
            return False
    
    def test_calibration(self) -> bool:
        """Test calibration functionality."""
        print("\nTesting Calibration System")
        print("-" * 40)
        
        try:
            print("Starting calibration process...")
            print("You will see 5 calibration points. Click on each point while looking at the red circle.")
            print("Press 'q' during calibration to skip.")
            
            run_calibration(2.0, "demo_recordings/demo_calibration_data.json")
            print("Calibration completed")
            # Run calibration with shorter duration for demo
            # data = []
            
            # with WebcamClient(self.demo_config["webcam"]) as cam:
            #     width, height, _ = cam._actual_props()
                
            #     for i, (normalized_x, normalized_y) in enumerate(CALIB_POINTS):
            #         print(f"Look at point {i+1}/5: ({normalized_x:.1f}, {normalized_y:.1f})")
            #         start = time.time()
            #         samples = []
                    
            #         while time.time() - start < 2.0:  # 2 seconds per point for demo
            #             frame = cam.snapshot(processors=[P.flip_horizontal])
            #             iris_center = P.get_iris_center(frame)
                        
            #             # Draw calibration dot
            #             x = int(normalized_x * width)
            #             y = int(normalized_y * height)
            #             cv2.circle(frame, (x, y), 20, (0, 0, 255), -1)
            #             cv2.putText(frame, f"Point {i+1}/5", (x-50, y-30), 
            #                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
            #             if iris_center:
            #                 samples.append(iris_center)
            #                 cv2.circle(frame, (int(iris_center[0]), int(iris_center[1])), 5, (0, 255, 255), -1)
                        
            #             cv2.imshow("Calibration", frame)
            #             if cv2.waitKey(1) & 0xFF == ord('q'):
            #                 break
                    
            #         if samples:
            #             avg = np.mean(samples, axis=0).tolist()
            #             data.append({
            #                 "screen_x": normalized_x,
            #                 "screen_y": normalized_y,
            #                 "eye_x": avg[0],
            #                 "eye_y": avg[1]
            #             })
            #             print(f"Captured point {i+1}: ({normalized_x}, {normalized_y}) -> eye {avg}")
                
            #     cv2.destroyAllWindows()
            
            # if len(data) >= 3:  # Need at least 3 points for basic calibration
            #     self.save_calibration_data(data)
            #     print(f"Calibration completed with {len(data)} points")
            #     return True
            # else:
            #     print("Insufficient calibration points captured")
            #     return False
                
        except Exception as e:
            print(f"Calibration test failed: {e}")
            return False
    
    def test_eye_tracking(self) -> bool:
        """Test eye tracking functionality using calibration data."""
        print("Running calibrateion before eye tracking is required")
        print("-" * 40)
        
        if not self.calibration_data:
            print("No calibration data available. Run calibration first.")
            return False
        
        try:
            print("Starting eye tracking...")
            print("Press 'q' to quit")
            run_eye_tracking()
            print("Eye tracking test completed")
            return True
            
        except Exception as e:
            print(f"Eye tracking test failed: {e}")
            return False
    
    def test_screen_recording(self) -> bool:
        """Test screen recording functionality."""
        print("\nTesting Screen Recording")
        print("-" * 40)
        
        try:
            recorder = ScreenRecorder(self.demo_config["screen_recording"])
            print("Starting 5-second screen recording...")
            print("Move your mouse around or open some windows to see activity")
            
            video_path = recorder.record(duration_seconds=5.0)
            
            if video_path.exists():
                file_size = video_path.stat().st_size
                print(f"Screen recording completed: {video_path.name} ({file_size} bytes)")
                return True
            else:
                print("Screen recording failed - no file created")
                return False
                
        except Exception as e:
            print(f"Screen recording test failed: {e}")
            return False
    
    def test_integrated_demo(self) -> bool:
        """Test integrated functionality - gaze tracking + screen recording."""
        print("\nTesting Integrated Demo (Gaze + Screen Recording)")
        print("-" * 40)
        
        if not self.calibration_data:
            print("No calibration data available. Run calibration first.")
            return False
        
        try:
            # Start screen recording
            recorder = ScreenRecorder(self.demo_config["screen_recording"])
            video_path = recorder.start()
            print(f"Screen recording started: {video_path.name}")
            
            # Start gaze tracking
            with WebcamClient(self.demo_config["webcam"]) as cam:
                print("Starting integrated gaze tracking...")
                print("Look around while screen is being recorded")
                print("Press 'q' to stop early")
                
                start_time = time.time()
                gaze_data = []
                
                while time.time() - start_time < 8:  # 8 second demo
                    frame = cam.snapshot(processors=[P.flip_horizontal])
                    iris_center = P.get_iris_center(frame)
                    
                    if iris_center:
                        # Store gaze data with timestamp
                        gaze_data.append({
                            "timestamp": time.time(),
                            "iris_x": iris_center[0],
                            "iris_y": iris_center[1]
                        })
                        
                        # Draw iris center
                        cv2.circle(frame, (int(iris_center[0]), int(iris_center[1])), 5, (0, 255, 0), -1)
                        cv2.putText(frame, f"Gaze: ({iris_center[0]:.1f}, {iris_center[1]:.1f})", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.putText(frame, "Integrated Demo - Screen Recording Active", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    cv2.imshow("Integrated Demo", frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                cv2.destroyAllWindows()
            
            # Stop screen recording
            recorder.stop()
            
            # Save gaze data
            gaze_file = self.recordings_dir / f"gaze_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(gaze_file, 'w') as f:
                json.dump(gaze_data, f, indent=2)
            
            print(f"Integrated demo completed")
            print(f"   Screen recording: {video_path.name}")
            print(f"   Gaze data: {gaze_file.name} ({len(gaze_data)} samples)")
            
            return True
            
        except Exception as e:
            print(f"Integrated demo failed: {e}")
            return False
    
    def show_menu(self) -> None:
        """Display the demo menu."""
        print("\nPixelsToPlayers Demo Menu")
        print("=" * 30)
        print("1. Test Webcam (Face Mesh Detection)")
        print("2. Test iris Tracking")
        print("3. Run Calibration")
        print("4. Run eye-to-screen Tracking")
        print("5. Test Screen Recording")
        print("6. Integrated Demo (Gaze + Screen Recording)")
        print("7. Run All Tests")
        print("0. Exit")
        print("-" * 30)
    
    def run_all_tests(self) -> None:
        """Run all tests in sequence."""
        print("\nRunning All Tests")
        print("=" * 30)
        
        tests = [
            ("Webcam Test", self.test_webcam_basic),
            ("Gaze Tracking Test", self.test_gaze_tracking),
            ("Calibration Test", self.test_calibration),
            ("Screen Recording Test", self.test_screen_recording),
            ("Integrated Demo", self.test_integrated_demo)
        ]
        
        results = []
        for name, test_func in tests:
            print(f"\nRunning {name}...")
            try:
                result = test_func()
                results.append((name, result))
                if result:
                    print(f"{name} PASSED")
                else:
                    print(f"{name} FAILED")
            except Exception as e:
                print(f"{name} ERROR: {e}")
                results.append((name, False))
        
        # Summary
        print("\n" + "=" * 30)
        print("Test Results Summary")
        print("-" * 30)
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for name, result in results:
            status = "PASS" if result else "FAIL"
            print(f"{name}: {status}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("All tests passed!")
        else:
            print(f"{total - passed} test(s) failed. Check the output above.")
    
    def run_demo(self) -> None:
        """Main demo loop."""
        print("Welcome to PixelsToPlayers Demo!")
        print("This prototype demonstrates all the components built so far.")
        print("This demo needs camera and recording permissions so please start any test, allow permissions, and restart if this is your first time running the demo")
        print("Follow along in the terminal, some instructions might have you tab in to the application that opens")

        
        # Try to load existing calibration data
        self.load_calibration_data()
        
        while True:
            self.show_menu()
            
            try:
                choice = input("\nEnter your choice (0-7): ").strip()
                
                if choice == "0":
                    print("Goodbye!")
                    break
                elif choice == "1":
                    self.test_webcam_basic()
                elif choice == "2":
                    self.test_gaze_tracking()
                elif choice == "3":
                    self.test_calibration()
                elif choice == "4":
                    self.test_eye_tracking()
                elif choice == "5":
                    self.test_screen_recording()
                elif choice == "6":
                    self.test_integrated_demo()
                elif choice == "7":
                    self.run_all_tests()
                else:
                    print("Invalid choice. Please enter 0-7.")
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\nDemo interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                input("Press Enter to continue...")


def main():
    """Main entry point for the demo."""
    try:
        demo = PixelsToPlayersDemo()
        demo.run_demo()
    except Exception as e:
        print(f"Demo failed to start: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())