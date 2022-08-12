def progress_bar(progress, total, display_length=60):
        left_ratio = display_length * progress//total
        right_ratio = display_length - left_ratio
        
        print('['+ '='*left_ratio + '>' + '.'*right_ratio + f'] {progress} / {total}', end='\r') 