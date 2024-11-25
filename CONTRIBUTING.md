
# Contributing to Vital_SQI

🎉 Thank you for your interest in contributing to **Vital_SQI**! 🎉  
We’re thrilled to have you on board. Below is a step-by-step guide to help you navigate the contribution process.

---

## How Can You Contribute? 🤔

There are several ways you can contribute to the project:
- **Report Bugs** 🐛
- **Request Features** ✨
- **Submit Pull Requests** 🔧
- **Improve Documentation** 📖

---

## Reporting Issues 🐞

If you find a bug or want to suggest an enhancement, let us know:
1. Go to the [Issues](https://github.com/Oucru-Innovations/vital-sqi/issues) page.
2. Click on `New Issue`.
3. Choose the appropriate template:
   - **Bug Report** for bugs.
   - **Feature Request** for enhancements.
4. Fill out the form and submit!

---

## Contributing Code 🔨

Ready to dive into the code? Awesome! Here’s how you can get started:

### 1. Fork the Repository 🍴
Click the **Fork** button at the top right of the [Vital_SQI GitHub page](https://github.com/Oucru-Innovations/vital-sqi).

### 2. Clone Your Fork 🖥️
```bash
git clone https://github.com/<your-username>/vital-sqi.git
cd vital-sqi
```

### 3. Create a New Branch 🌿
Always create a new branch for your work:
```bash
git checkout -b feature/my-awesome-feature
```

### 4. Install Dependencies 📦
We use `pip` for dependencies:
```bash
pip install -r requirements.txt
```

### 5. Make Your Changes ✏️
- Write clean, modular, and well-documented code.
- Ensure your changes align with our [Code of Conduct](#code-of-conduct).

### 6. Run Tests ✅
Before pushing your code, ensure all tests pass:
```bash
pytest tests/
```

### 7. Push Your Changes 🚀
```bash
git add .
git commit -m "Add my awesome feature"
git push origin feature/my-awesome-feature
```

### 8. Open a Pull Request 🔃
1. Go to the [Pull Requests](https://github.com/Oucru-Innovations/vital-sqi/pulls) page.
2. Click `New Pull Request`.
3. Choose your branch and describe your changes.

---

## Style Guide 🖌️

To keep our code consistent:
- Follow **PEP8** guidelines.
- Use meaningful variable and function names.
- Document your code with docstrings.
- Format code using `flake8`:
```bash
flake8 --config=.flake8 vital_sqi tests
```

---

## Local Development 🛠️

For development and debugging, use the `Makefile`:
- Install the package:
  ```bash
  make install
  ```
- Run tests with coverage:
  ```bash
  make test
  ```

---

## Code of Conduct ❤️

Be respectful and kind. We are committed to fostering an inclusive community where everyone feels welcome.

---

## Need Help? 🤝

If you’re stuck or have any questions, feel free to:
- Open a discussion on the [Discussions](https://github.com/Oucru-Innovations/vital-sqi/discussions) page.
- Reach out via email: **support@oucru-innovations.org**.

---

Thank you for contributing! Your efforts make **Vital_SQI** better for everyone. 🚀

— The **Vital_SQI** Development Team
