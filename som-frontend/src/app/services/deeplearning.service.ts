import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class DeeplearningService {

  private apiUrl = 'http://localhost:5000/dl';

  constructor(private http: HttpClient) { }

  trainDeepLearning(): Observable<any> {
    return this.http.post(`${this.apiUrl}/train`, {});
  }

  getImage(type: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/image/${type}`, { responseType: 'blob' });
  }

  getAccuracy(type: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/accuracy/${type}`);
  }
}