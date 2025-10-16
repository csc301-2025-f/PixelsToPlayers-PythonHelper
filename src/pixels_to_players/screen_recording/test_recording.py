#!/usr/bin/env python3
"""
Simple test for screen recording functionality
"""

import time
from pathlib import Path
from recording import ScreenRecorder, ScreenRecordingConfig

def test_basic_recording():
    """Test basic recording without preview"""
    print("Test 1: Basic recording (no preview)")
    try:
        config = ScreenRecordingConfig(
            fps=15,
            show_preview=False,
            enable_key_stop=False
        )
        recorder = ScreenRecorder(config)
        video_path = recorder.record(duration_seconds=5.0)
        
        if video_path.exists():
            file_size = video_path.stat().st_size
            print(f"PASS: Basic recording worked - {video_path.name} ({file_size} bytes)")
            print(f"  This should be a clean recording without any preview window interference")
            return True
        else:
            print("FAIL: Basic recording - file not created")
            return False
    except Exception as e:
        print(f"FAIL: Basic recording - {e}")
        return False

def test_gaming_mode():
    """Test that ESC and q keys don't stop recording"""
    print("\nTest 2: Gaming mode (ESC and q keys should NOT stop recording)")
    try:
        config = ScreenRecordingConfig(
            fps=10,
            show_preview=True,
            enable_key_stop=False
        )
        recorder = ScreenRecorder(config)
        print("Recording for 5 seconds... try pressing ESC or q - it should keep recording")
        video_path = recorder.record(duration_seconds=5.0)
        
        if video_path.exists():
            file_size = video_path.stat().st_size
            print(f"PASS: Gaming mode worked - {video_path.name} ({file_size} bytes)")
            return True
        else:
            print("FAIL: Gaming mode - file not created")
            return False
    except Exception as e:
        print(f"FAIL: Gaming mode - {e}")
        return False

def test_key_stop_mode():
    """Test that ESC and q keys DO stop recording"""
    print("\nTest 3: Key stop mode (ESC and q keys SHOULD stop recording)")
    try:
        config = ScreenRecordingConfig(
            fps=10,
            show_preview=True,
            enable_key_stop=True
        )
        recorder = ScreenRecorder(config)
        print("Recording for 8 seconds... press ESC or q to stop early")
        video_path = recorder.record(duration_seconds=8.0)
        
        if video_path.exists():
            file_size = video_path.stat().st_size
            print(f"PASS: Key stop mode worked - {video_path.name} ({file_size} bytes)")
            return True
        else:
            print("FAIL: Key stop mode - file not created")
            return False
    except Exception as e:
        print(f"FAIL: Key stop mode - {e}")
        return False

def test_manual_control():
    """Test manual start/stop"""
    print("\nTest 4: Manual start/stop")
    try:
        config = ScreenRecordingConfig(
            fps=10,
            show_preview=False,
            enable_key_stop=False
        )
        recorder = ScreenRecorder(config)
        
        video_path = recorder.start()
        print(f"Started recording: {video_path.name}")
        
        time.sleep(3.0)
        
        recorder.stop()
        
        if video_path.exists():
            file_size = video_path.stat().st_size
            print(f"PASS: Manual control worked - {video_path.name} ({file_size} bytes)")
            return True
        else:
            print("FAIL: Manual control - file not created")
            return False
    except Exception as e:
        print(f"FAIL: Manual control - {e}")
        return False

def cleanup_recordings():
    """Delete test recordings"""
    print("\nCleaning up test recordings...")
    recordings_dir = Path("recordings")
    if recordings_dir.exists():
        for file in recordings_dir.glob("screen_*.mp4"):
            try:
                file.unlink()
                print(f"Deleted: {file.name}")
            except Exception as e:
                print(f"Couldn't delete {file.name}: {e}")

def main():
    """Run the tests"""
    print("Screen Recording Test")
    print("=" * 30)
    
    tests = [
        test_basic_recording,
        test_gaming_mode,
        test_key_stop_mode,
        test_manual_control
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 30)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! Screen recording is working.")
    else:
        print(f"{total - passed} test(s) failed.")
    
    try:
        response = input("\nClean up test recordings? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            cleanup_recordings()
        else:
            print("Test recordings kept in: recordings/")
    except KeyboardInterrupt:
        print("\nTest recordings kept in: recordings/")

if __name__ == "__main__":
    main()
