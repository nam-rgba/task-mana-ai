import json
import re
# File hỗ trợ gán thêm type cho mock dữ liệu
# Đọc file tasks.json
with open('ai_service/app/data/tasks.json', 'r', encoding='utf-8') as f:
    tasks = json.load(f)

def determine_task_type(task):
    """Xác định type phù hợp cho task dựa trên title và description"""
    title = task.get('title', '').lower()
    description = task.get('description', '').lower()
    combined = title + ' ' + description
    
    # FEATURE - các tính năng mới
    feature_keywords = ['xây dựng', 'phát triển', 'tạo tính năng', 'thiết kế hệ thống', 
                       'tích hợp', 'thêm hệ thống', 'module', 'chức năng', 
                       'xây dựng cơ chế', 'viết logic ai']
    
    # BUG - sửa lỗi
    bug_keywords = ['sửa lỗi', 'fix bug', 'khắc phục', 'debug']
    
    # IMPROVEMENT - cải thiện, tối ưu
    improvement_keywords = ['tối ưu', 'cải thiện', 'nâng cấp', 'optimize']
    
    # RESEARCH - nghiên cứu, phân tích
    research_keywords = ['nghiên cứu', 'phân tích', 'thu thập', 'khảo sát', 
                        'xác định phạm vi', 'research']
    
    # DOCUMENTATION - tài liệu
    documentation_keywords = ['tài liệu', 'documentation', 'hướng dẫn', 'release notes',
                             'chuẩn hoá', 'đóng gói']
    
    # TESTING - kiểm thử
    testing_keywords = ['kiểm thử', 'test', 'đánh giá chất lượng', 'viết test case',
                       'chạy kiểm thử', 'regression', 'qa']
    
    # DEPLOYMENT - triển khai
    deployment_keywords = ['triển khai', 'deploy', 'môi trường chính thức', 'release']
    
    # ENHANCEMENT - cải tiến thiết kế
    enhancement_keywords = ['thiết kế', 'design', 'kiến trúc', 'interface', 
                           'prototype', 'mẫu thiết kế', 'state machine', 'flow']
    
    # MAINTENANCE - bảo trì
    maintenance_keywords = ['bảo trì', 'maintain', 'hoàn thiện', 'vá lỗi']
    
    # Kiểm tra theo thứ tự ưu tiên
    if any(keyword in combined for keyword in bug_keywords):
        return 'BUG'
    
    if any(keyword in combined for keyword in testing_keywords):
        return 'TESTING'
    
    if any(keyword in combined for keyword in documentation_keywords):
        return 'DOCUMENTATION'
    
    if any(keyword in combined for keyword in deployment_keywords):
        return 'DEPLOYMENT'
    
    if any(keyword in combined for keyword in research_keywords):
        return 'RESEARCH'
    
    if any(keyword in combined for keyword in improvement_keywords):
        return 'IMPROVEMENT'
    
    if any(keyword in combined for keyword in enhancement_keywords):
        return 'ENHANCEMENT'
    
    if any(keyword in combined for keyword in maintenance_keywords):
        return 'MAINTENANCE'
    
    if any(keyword in combined for keyword in feature_keywords):
        return 'FEATURE'
    
    # Mặc định
    return 'OTHER'

# Thêm type cho mỗi task
for task in tasks:
    task['type'] = determine_task_type(task)

# Ghi lại file với type đã được thêm
with open('ai_service/app/data/tasks.json', 'w', encoding='utf-8') as f:
    json.dump(tasks, f, ensure_ascii=False, indent=2)

print(f"✓ Đã cập nhật type cho {len(tasks)} tasks thành công!")

# Thống kê
type_counts = {}
for task in tasks:
    task_type = task['type']
    type_counts[task_type] = type_counts.get(task_type, 0) + 1

print("\nThống kê số lượng tasks theo type:")
for task_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {task_type}: {count} tasks")
