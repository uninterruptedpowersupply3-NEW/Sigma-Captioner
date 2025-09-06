# main.py
# Application entry point. Initializes and displays the GUI.

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont # Import QFont for global font setting
from gui import MainWindow

if __name__ == '__main__':
    # Initialize the PyQt Application
    app = QApplication(sys.argv)
    
    # Set a modern style sheet for a better look and feel
    app.setStyle('Fusion')
    
    # Set a default font for the entire application for better aesthetics
    # Using 'Inter' if available, otherwise a common sans-serif font
    font = QFont("Inter", 9)
    if not font.fromString("Inter"): # Check if 'Inter' is available
        font = QFont("Segoe UI", 9) # Fallback for Windows
        if not font.fromString("Segoe UI"):
            font = QFont("Arial", 9) # Generic fallback
    app.setFont(font)

    # Create and show the main window
    window = MainWindow()
    window.show()
    
    # Start the application event loop
    sys.exit(app.exec())