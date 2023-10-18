# Use an official Python 3.11.5 runtime as a parent image
FROM python:3.11.5

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define the command to run your application
CMD ["python3", "app.py"]
