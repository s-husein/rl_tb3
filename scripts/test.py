class MyClass:
    def my_decorator(self, func):
        def wrapper(*args, **kwargs):
            print("Something is happening before the method is called.")
            result = func(*args, **kwargs)
            print("Something is happening after the method is called.")
            return result
        return wrapper

    @my_decorator  # Applying the decorator to a method
    def my_method(self):
        print("Executing the method.")

# Create an instance of MyClass
obj = MyClass()

# Call the decorated method
obj.my_method()