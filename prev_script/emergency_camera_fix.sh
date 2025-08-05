#!/bin/bash
# Emergency camera diagnostic script for Raspberry Pi
# This script focuses on identifying hardware-level issues

echo "==== Emergency Camera Diagnostics ===="

# Check USB devices
echo "1. Checking USB devices..."
echo "USB devices connected:"
lsusb
echo ""

# Check for camera module
echo "2. Checking for Raspberry Pi camera module..."
if vcgencmd get_camera | grep -q "supported=1 detected=1"; then
    echo "✓ Raspberry Pi camera module detected"
else
    echo "✗ Raspberry Pi camera module not detected or not enabled"
    echo "  This is normal if you're using a USB webcam instead"
fi
echo ""

# Check power supply
echo "3. Checking power supply status..."
vcgencmd get_throttled
echo "  Note: If value is not 0x0, there may be power issues"
echo ""

# Check USB power
echo "4. Checking USB power settings..."
cat /sys/module/usbcore/parameters/autosuspend || echo "Could not check USB autosuspend"
echo ""

# Check for kernel messages related to USB/camera
echo "5. Checking kernel messages related to camera/USB..."
dmesg | grep -E "usb|camera|video" | tail -20
echo ""

# Install critical tools if needed
echo "6. Installing diagnostic tools..."
sudo apt update
sudo apt install -y v4l-utils fswebcam
echo ""

# Try to list video devices and formats with v4l2-ctl
echo "7. Listing video devices and formats..."
for device in /dev/video*; do
    if [ -e "$device" ]; then
        echo "Device: $device"
        v4l2-ctl --device=$device --info
        v4l2-ctl --device=$device --list-formats-ext
        echo ""
    fi
done

# Try simple direct capture with dd
echo "8. Attempting direct device read..."
for device in /dev/video*; do
    if [ -e "$device" ]; then
        echo "Testing direct read from $device..."
        timeout 2s dd if=$device of=/dev/null bs=1M count=1 2>&1 || echo "Failed to read directly from $device"
    fi
done
echo ""

# Try multiple fswebcam settings
echo "9. Testing camera with multiple fswebcam settings..."
for device in /dev/video*; do
    if [ -e "$device" ]; then
        echo "Testing fswebcam with $device..."
        fswebcam -d $device -r 320x240 --no-banner ~/jaundice_monitor/fswebcam_test_${device##*/}_1.jpg
        fswebcam -d $device -r 640x480 --no-banner ~/jaundice_monitor/fswebcam_test_${device##*/}_2.jpg
        # Try with different input format
        fswebcam -d $device -r 320x240 --no-banner -p YUYV ~/jaundice_monitor/fswebcam_test_${device##*/}_3.jpg
    fi
done

# Check saved images
echo "10. Checking if any test images were captured..."
ls -la ~/jaundice_monitor/fswebcam_test_*.jpg 2>/dev/null || echo "No test images were captured"
echo ""

# Try to run camera with motion (different approach)
echo "11. Checking if motion can access the camera..."
sudo apt install -y motion
sudo systemctl stop motion
sleep 2
sudo motion -n -c /etc/motion/motion.conf &
MOTION_PID=$!
echo "Started motion with PID $MOTION_PID"
sleep 10
kill $MOTION_PID
echo "Stopped motion"
echo "Check if images were saved to /var/lib/motion/"
ls -la /var/lib/motion/ 2>/dev/null || echo "No motion images found"
echo ""

echo "==== USB Power Management ===="
echo "Disabling USB autosuspend (may help with power issues)..."
echo -1 | sudo tee /sys/module/usbcore/parameters/autosuspend

echo "==== Kernel Module Check ===="
echo "Loading video modules that might help..."
sudo modprobe uvcvideo
sudo modprobe videodev
sudo modprobe v4l2_common

echo ""
echo "==== EMERGENCY FIXES ===="
echo "1. Try a different USB port (especially one closer to the power input)"
echo "2. Use a powered USB hub"
echo "3. Update Raspberry Pi firmware: sudo rpi-update"
echo "4. Make sure your power supply provides sufficient power (2.5A+ recommended)"
echo "5. Try a different USB webcam"
echo ""
echo "To check if your camera is simply incompatible with Raspberry Pi,"
echo "please share the output of 'lsusb' and the camera model from above."
